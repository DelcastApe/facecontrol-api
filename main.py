# FastAPI Core
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Body, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Librerías externas
import face_recognition
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock
from PIL import Image
from fpdf import FPDF
from jose import JWTError, jwt
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
import csv

# Utilidades del sistema
import os
import io
import json
from uuid import uuid4
import random
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

# Supabase
from supabase import create_client
from supabase_client import supabase

# Funciones propias
from utils.seguridad import crear_token, verificar_token, verificar_token_general


load_dotenv()


ADMIN_ID = os.getenv("ADMIN_ID")
SECRET_KEY = os.getenv("SECRET_KEY") 
ALGORITHM = "HS256"
BUCKET_NAME = os.getenv("BUCKET_NAME", "rostros")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


app = FastAPI()


# CORS habilitado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "🚀 API corriendo correctamente"}

@app.get("/reconocimientos")
def get_reconocimientos():
    try:
        response = supabase.table("reconocimientos") \
            .select("persona_id, fecha, hora, personas(nombre, apellidos)") \
            .order("fecha", desc=True) \
            .order("hora", desc=True) \
            .limit(10) \
            .execute()

        datos = []
        for a in response.data:
            datos.append({
                "nombre": a["personas"]["nombre"],
                "apellidos": a["personas"]["apellidos"],
                "timestamp": f'{a["fecha"]} {a["hora"]}'
            })
        return datos
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




def ya_fue_reconocido_recientemente(persona_id: str) -> bool:
    ahora = datetime.now()
    fecha_actual = ahora.date().isoformat()
    hora_limite = (ahora - timedelta(hours=1)).time().isoformat()
    try:
        reconocimientos = supabase.table("reconocimientos") \
            .select("hora") \
            .eq("persona_id", persona_id) \
            .eq("fecha", fecha_actual) \
            .execute()
        for registro in reconocimientos.data:
            if registro["hora"] > hora_limite:
                return True
        return False
    except:
        return False

def extraer_embedding(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_np = np.array(img)

    face_locations = face_recognition.face_locations(img_np)
    if not face_locations:
        raise ValueError("No se detectó ningún rostro en la imagen.")
    
    embeddings = face_recognition.face_encodings(img_np, face_locations)
    if not embeddings:
        raise ValueError("No se pudieron extraer las características del rostro.")
    
    return embeddings[0].tolist()  # Vector de 128 valores

def score_similitud_hibrida(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # Distancias
    cos_sim = 1 - cosine(vec1, vec2)  # Similitud del coseno (1 - distancia)
    euc_dist = euclidean(vec1, vec2)
    l1_dist = cityblock(vec1, vec2)

    # Normalización simple de distancias para el score final
    score = (cos_sim * 0.6) + ((1 / (1 + euc_dist)) * 0.2) + ((1 / (1 + l1_dist)) * 0.2)
    return score


@app.post("/registrar_persona")
async def registrar_persona(
    file: UploadFile = File(...),
    nombre: str = Form(...),
    apellidos: str = Form(...),
    correo: str = Form(...),  # ← AHORA SÍ se captura el campo correo
    requisitoriado: bool = Form(...)
):
    try:
        contents = await file.read()
        embedding = extraer_embedding(contents)

        file_name = f"{uuid.uuid4()}_{file.filename}"
        upload_result = supabase.storage.from_("rostros").upload(file_name, contents, {"content-type": file.content_type})

        # Validación de subida
        if hasattr(upload_result, "data") and upload_result.data is None:
            return JSONResponse(status_code=500, content={"error": "Error al subir imagen al bucket Supabase."})

        # URL pública
        public_url_response = supabase.storage.from_("rostros").get_public_url(file_name)
        foto_url = public_url_response.get("publicUrl") if isinstance(public_url_response, dict) else public_url_response

        # Registro en BD
        response_db = supabase.table("personas").insert({
            "nombre": nombre,
            "apellidos": apellidos,
            "correo": correo,
            "kp": embedding,
            "foto": foto_url,
            "requisitoriado": requisitoriado
        }).execute()

        return {"message": "✅ Persona registrada exitosamente.", "persona_id": response_db.data[0]["id"]}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})





def enviar_correo_alerta(persona, file_bytes):
    try:
        cuerpo = (
            f"🚨 ALERTA: Persona requisitoriada detectada\n\n"
            f"Nombre: {persona['nombre']} {persona['apellidos']}\n"
            f"ID: {persona['id']}\n"
            f"Score: {persona['score']}"
        )

        # Construcción del correo
        msg = MIMEMultipart()
        msg['Subject'] = '🚨 ALERTA DE SEGURIDAD'
        msg['From'] = os.getenv("SMTP_USER")
        msg['To'] = os.getenv("ALERTA_DESTINO_MAIL")

        # Cuerpo del mensaje
        msg.attach(MIMEText(cuerpo, 'plain'))

        # Adjuntar la imagen como .jpg
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file_bytes)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="rostro_detectado.jpg"')
        msg.attach(part)

        # Enviar
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
        server.send_message(msg)
        server.quit()

        print("✅ Correo enviado correctamente con imagen")
    except Exception as e:
        print("❌ Error al enviar correo:", e)





def enviar_sms_alerta(persona):
    try:
        client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_AUTH"))
        mensaje = (
            f"🚨 ALERTA: {persona['nombre']} {persona['apellidos']} fue detectado como REQUISITORIADO."
        )

        message = client.messages.create(
            body=mensaje,
            from_=os.getenv("TWILIO_PHONE"),
            to=os.getenv("ALERTA_DESTINO_SMS")
        )
        print("✅ SMS enviado correctamente")
    except Exception as e:
        print("❌ Error al enviar SMS:", e)





@app.post("/reconocer")
async def reconocer_rostro(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    latitud: float = Form(None),
    longitud: float = Form(None)
):
    try:
        contents = await file.read()
        encoding_actual = extraer_embedding(contents)

        personas = supabase.table("personas").select("id, nombre, apellidos, kp, requisitoriado").execute()
        matches = []

        for persona in personas.data:
            if not persona["kp"]:
                continue

            score = score_similitud_hibrida(encoding_actual, persona["kp"])

            if score > 0.75:
                ahora = datetime.now()

                print(f"\n[DEBUG] Coincidencia: {persona['nombre']} {persona['apellidos']} | Score: {round(score, 3)}")

                if not ya_fue_reconocido_recientemente(persona["id"]):
                    lat = latitud if latitud is not None else round(random.uniform(-9.1, -8.0), 6)
                    lon = longitud if longitud is not None else round(random.uniform(-79.1, -77.0), 6)

                    print(f"[DEBUG] Insertando en reconocimientos -> latitud: {lat}, longitud: {lon}")

                    resp_reco = supabase.table("reconocimientos").insert({
                        "persona_id": persona["id"],
                        "fecha": ahora.date().isoformat(),
                        "hora": ahora.time().strftime("%H:%M:%S"),
                        "latitud": lat,
                        "longitud": lon
                    }).execute()
                    print(f"[DEBUG] Supabase resp_reco: {resp_reco}")

                # Entrenamiento adaptativo
                print("[DEBUG] Insertando en entrenamiento")
                resp_entrena = supabase.table("entrenamientos").insert({
                    "persona_id": persona["id"],
                    "kp": json.loads(json.dumps(encoding_actual)),
                    "fecha": ahora.date().isoformat(),
                    "hora": ahora.time().strftime("%H:%M:%S")
                }).execute()
                print(f"[DEBUG] Supabase resp_entrena: {resp_entrena}")

                entrenamientos = supabase.table("entrenamientos") \
                    .select("kp") \
                    .eq("persona_id", persona["id"]) \
                    .limit(10) \
                    .execute()

                if len(entrenamientos.data) == 10:
                    print("[DEBUG] Promediando KP de entrenamiento")
                    vectores = [np.array(e["kp"]) for e in entrenamientos.data]
                    promedio = np.mean(vectores, axis=0).tolist()

                    supabase.table("personas").update({"kp": promedio}).eq("id", persona["id"]).execute()
                    supabase.table("entrenamientos").delete().eq("persona_id", persona["id"]).execute()

                # 🔐 Token
                token = crear_token({"sub": persona["id"]})

                match_info = {
                    "id": persona["id"],
                    "nombre": persona["nombre"],
                    "apellidos": persona["apellidos"],
                    "requisitoriado": persona["requisitoriado"],
                    "score": round(score, 3),
                    "is_admin": persona["id"] == ADMIN_ID,
                    "token": token
                }

                matches.append(match_info)

                if persona["requisitoriado"]:
                    print(f"\n🚨 ALERTA DE SEGURIDAD -> Persona requisitoriada: {match_info}")
                    background_tasks.add_task(enviar_correo_alerta, match_info, contents)
                    background_tasks.add_task(enviar_sms_alerta, match_info)

                    print("[DEBUG] Insertando en alertas")
                    resp_alerta = supabase.table("alertas").insert({
                        "persona_id": persona["id"],
                        "nombre": persona["nombre"],
                        "apellidos": persona["apellidos"],
                        "score": round(score, 3),
                        "fecha": ahora.date().isoformat(),
                        "hora": ahora.time().strftime("%H:%M:%S"),
                        "metodo_envio": "ambos"
                    }).execute()
                    print(f"[DEBUG] Supabase resp_alerta: {resp_alerta}")

        if matches:
            return {"message": "✅ Rostro reconocido", "coincidencias": matches}
        else:
            return {"message": "❌ Rostro no reconocido"}

    except Exception as e:
        print(f"❌ ERROR FATAL EN /reconocer: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})






@app.get("/mapa-reconocimientos")
def get_mapa_reconocimientos(user_id: str = Depends(verificar_token)):
    try:
        response = supabase.table("reconocimientos") \
            .select("persona_id, latitud, longitud, fecha, hora, personas(nombre, apellidos)") \
            .order("fecha", desc=True) \
            .order("hora", desc=True) \
            .limit(100) \
            .execute()

        resultados = []
        for registro in response.data:
            resultados.append({
                "nombre": registro["personas"]["nombre"],
                "apellidos": registro["personas"]["apellidos"],
                "latitud": registro["latitud"],
                "longitud": registro["longitud"],
                "timestamp": f'{registro["fecha"]} {registro["hora"]}'
            })

        return resultados
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})






@app.get("/personas")
def listar_personas(user_id: str = Depends(verificar_token)):
    try:
        # Verificamos si es el admin
        is_admin = user_id == ADMIN_ID

        if is_admin:
            response = supabase.table("personas").select("*").order("nombre", desc=True).execute()
        else:
            # Un usuario solo puede verse a sí mismo
            response = supabase.table("personas").select("*").eq("id", user_id).execute()

        return {"personas": response.data}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})





@app.put("/admin/personas/{persona_id}")
async def actualizar_persona(
    persona_id: str,
    nombre: str = Form(...),
    apellidos: str = Form(...),
    correo: str = Form(...),
    requisitoriado: bool = Form(...),
    file: UploadFile = File(None),
):
    try:
        nueva_url = None

        if file:
            extension = file.filename.split('.')[-1]
            nombre_archivo = f"{str(uuid4())}.{extension}"
            contenido = await file.read()

            subir = supabase.storage.from_("rostros").upload(
                nombre_archivo,
                contenido,
                {"content-type": file.content_type}
            )

            # ✅ Validar si hubo error en la subida
            if hasattr(subir, "error") and subir.error is not None:
                print("❌ Error al subir imagen:", subir.error)
                raise HTTPException(status_code=500, detail="Error al subir imagen a Supabase.")

            # ✅ Obtener URL pública
            public_url_response = supabase.storage.from_("rostros").get_public_url(nombre_archivo)
            nueva_url = public_url_response.get("publicUrl") if isinstance(public_url_response, dict) else public_url_response
            print(f"✅ Imagen subida correctamente. URL: {nueva_url}")
        else:
            print("ℹ️ No se envió una nueva imagen. Se mantiene la foto anterior.")

        # Datos a actualizar
        datos_actualizados = {
            "nombre": nombre,
            "apellidos": apellidos,
            "correo": correo,
            "requisitoriado": requisitoriado
        }

        if nueva_url:
            datos_actualizados["foto"] = nueva_url

        actualizacion = supabase.table("personas").update(datos_actualizados).eq("id", persona_id).execute()

        return {"mensaje": "✅ Persona actualizada correctamente", "persona": actualizacion.data}

    except Exception as e:
        print("❌ [ERROR] Error al actualizar persona:", e)
        raise HTTPException(status_code=500, detail=f"Error al actualizar persona: {e}")










@app.put("/perfil/editar")
async def editar_mi_perfil(
    datos: dict = Body(...),
    user_id: str = Depends(verificar_token_general)
):
    try:
        campos = {}
        for campo in ['nombre', 'apellidos', 'correo']:
            if campo in datos and datos[campo]:
                campos[campo] = datos[campo]

        if not campos:
            return JSONResponse(status_code=400, content={"error": "⚠️ No se enviaron campos para actualizar."})

        supabase.table("personas").update(campos).eq("id", user_id).execute()
        return {"message": "✅ Perfil actualizado correctamente."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# GET /perfil
@app.get("/perfil")
async def obtener_mi_perfil(user_id: str = Depends(verificar_token_general)):
    try:
        datos = supabase.table("personas").select("nombre, apellidos, correo, requisitoriado, foto").eq("id", user_id).single().execute()
        return datos.data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




# ELIMINAR PERSONA ADMIN
@app.delete("/personas/{persona_id}")
async def eliminar_persona(
    persona_id: str,
    user_id: str = Depends(verificar_token)
):
    try:
        if user_id != ADMIN_ID:
            raise HTTPException(status_code=403, detail="❌ Acceso denegado. Solo el administrador puede eliminar personas.")

        supabase.table("personas").delete().eq("id", persona_id).execute()
        return {"message": "✅ Persona eliminada correctamente."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




@app.get("/alertas")
def ver_alertas(user_id: str = Depends(verificar_token)):  # 🔒 Protegido con JWT solo para admin
    try:
        alertas = supabase.table("alertas") \
            .select("fecha, hora, nombre, apellidos, score, metodo_envio") \
            .order("fecha", desc=True) \
            .order("hora", desc=True) \
            .limit(20) \
            .execute()

        return alertas.data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/exportar-pdf")
async def exportar_pdf(
    modo: str = Form(...),  # "todos", "top10", "hoy"
    user_id: str = Depends(verificar_token)
):
    if user_id != ADMIN_ID:
        raise HTTPException(status_code=403, detail="❌ Solo el administrador puede exportar reportes.")

    try:
        hoy = date.today().isoformat()

        if modo == "top10":
            alertas_query = supabase.table("alertas") \
                .select("fecha,hora,nombre,apellidos,score,metodo_envio") \
                .order("fecha", desc=True).order("hora", desc=True).limit(10)

            reconocimientos_query = supabase.table("reconocimientos") \
                .select("fecha,hora,personas(nombre,apellidos,requisitoriado)") \
                .order("fecha", desc=True).order("hora", desc=True).limit(10)

        elif modo == "hoy":
            alertas_query = supabase.table("alertas") \
                .select("fecha,hora,nombre,apellidos,score,metodo_envio") \
                .eq("fecha", hoy)

            reconocimientos_query = supabase.table("reconocimientos") \
                .select("fecha,hora,personas(nombre,apellidos,requisitoriado)") \
                .eq("fecha", hoy)

        else:  # modo == "todos"
            alertas_query = supabase.table("alertas") \
                .select("fecha,hora,nombre,apellidos,score,metodo_envio") \
                .order("fecha", desc=True).order("hora", desc=True)

            reconocimientos_query = supabase.table("reconocimientos") \
                .select("fecha,hora,personas(nombre,apellidos,requisitoriado)") \
                .order("fecha", desc=True).order("hora", desc=True)

        alertas = alertas_query.execute().data
        reconocimientos = reconocimientos_query.execute().data

        # Generar PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Reporte General FACEAPP", ln=True, align="C")

        def agregar_tabla(titulo, data, columnas):
            pdf.set_font("Arial", "B", 14)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, f"\n{titulo}", ln=True, fill=True)
            pdf.set_font("Arial", "B", 10)
            for col in columnas:
                pdf.cell(32, 8, col, border=1)
            pdf.ln()
            pdf.set_font("Arial", "", 10)
            for row in data:
                for col in columnas:
                    valor = str(row.get(col)) if col in row else str((row.get("personas") or {}).get(col, ""))
                    pdf.cell(32, 8, valor, border=1)
                pdf.ln()

        columnas_alertas = ["fecha", "hora", "nombre", "apellidos", "score", "metodo_envio"]
        columnas_reconocimientos = ["fecha", "hora", "nombre", "apellidos", "requisitoriado"]

        agregar_tabla("1. ALERTAS", alertas, columnas_alertas)
        agregar_tabla("2. RECONOCIMIENTOS", reconocimientos, columnas_reconocimientos)

        # Descargar e insertar el logo de UPAO en la esquina inferior derecha
        logo_url = "https://descubre.upao.edu.pe/img/upao_logo.png"
        logo_response = requests.get(logo_url)

        if logo_response.status_code == 200:
            with open("logo_upao.png", "wb") as f:
                f.write(logo_response.content)

            page_width = pdf.w
            page_height = pdf.h
            pdf.image("logo_upao.png", x=page_width - 45, y=page_height - 30, w=35)

        # Generar PDF en memoria
        pdf_output = pdf.output(dest='S').encode('latin-1')
        buffer = BytesIO(pdf_output)

        return StreamingResponse(buffer, media_type="application/pdf", headers={
            "Content-Disposition": f"attachment; filename=faceapp_reporte_{modo}.pdf"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/entrenar/nuevo")
async def entrenamiento_manual(
    persona_id: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(verificar_token)
):
    if user_id != ADMIN_ID:
        raise HTTPException(status_code=403, detail="❌ Solo el administrador puede entrenar manualmente.")

    try:
        contents = await file.read()
        encoding = extraer_embedding(contents)
        ahora = datetime.now()

        # Insertar en la tabla de entrenamientos
        supabase.table("entrenamientos").insert({
            "persona_id": persona_id,
            "kp": json.loads(json.dumps(encoding)),  # asegurar compatibilidad con JSONB
            "fecha": ahora.date().isoformat(),
            "hora": ahora.time().strftime("%H:%M:%S")
        }).execute()

        return {"message": "✅ Imagen registrada para entrenamiento manual"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    
@app.get("/dashboard/stats")
async def obtener_estadisticas(user_id: str = Depends(verificar_token)):
    if user_id != ADMIN_ID:
        raise HTTPException(status_code=403, detail="Solo el administrador puede acceder a las estadísticas.")

    try:
        # Total de personas
        total_personas = supabase.table("personas").select("id").execute().data
        total_personas_count = len(total_personas)

        # Total de reconocimientos
        total_reconocimientos = supabase.table("reconocimientos").select("id").execute().data
        total_reconocimientos_count = len(total_reconocimientos)

        # Top 3 más reconocidos
        top3_query = supabase.rpc("top_personas_reconocidas").execute().data  # Asegúrate de tener esa función en Supabase

        # Requisitoriados reconocidos (mínimo 1 vez)
        requisitoriados = supabase.table("reconocimientos") \
            .select("personas(requisitoriado)") \
            .neq("personas.requisitoriado", False) \
            .execute().data
        requisitoriados_detectados = len(requisitoriados)

        # Porcentaje requisitoriados
        porcentaje_requisitoriados = (
            f"{requisitoriados_detectados} de {total_personas_count} personas están requisitoriadas "
            f"({round((requisitoriados_detectados / total_personas_count) * 100, 2)}%)"
        ) if total_personas_count > 0 else "No hay personas registradas."

        # Persona más recientemente reconocida
        reciente = supabase.table("reconocimientos") \
            .select("fecha,hora,personas(nombre,apellidos)") \
            .order("fecha", desc=True).order("hora", desc=True) \
            .limit(1).execute().data
        persona_reciente = reciente[0] if reciente else {}

        return {
            "total_personas": total_personas_count,
            "total_reconocimientos": total_reconocimientos_count,
            "top_3": top3_query,
            "requisitoriados_reconocidos": requisitoriados_detectados,
            "porcentaje_requisitoriados": porcentaje_requisitoriados,
            "persona_mas_reciente": persona_reciente
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
