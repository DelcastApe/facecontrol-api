from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
from supabase_client import supabase # Asegúrate de que supabase_client.py esté en el mismo directorio o accesible
import io
from datetime import datetime, timedelta
import uuid # Para generar IDs únicos para las imágenes en Storage

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

# ✅ Endpoint para el dashboard de asistencias
@app.get("/asistencias")
def get_asistencias():
    try:
        response = supabase.table("asistencias") \
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
        print(f"Error al obtener asistencias: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Verifica si ya registró asistencia hace menos de una hora
def ya_tiene_asistencia_reciente(persona_id: str) -> bool:
    ahora = datetime.now()
    fecha_actual = ahora.date().isoformat()
    
    # Restamos una hora a la hora actual para definir el límite inferior
    hora_limite = (ahora - timedelta(hours=1)).time().isoformat()

    try:
        asistencias = supabase.table("asistencias") \
            .select("hora") \
            .eq("persona_id", persona_id) \
            .eq("fecha", fecha_actual) \
            .execute()
        
        for registro in asistencias.data:
            # Comparamos solo las horas para ver si está dentro de la última hora
            if registro["hora"] > hora_limite:
                return True
        return False
    except Exception as e:
        print(f"Error al verificar asistencia reciente: {e}")
        return False # En caso de error, asumimos que no tiene asistencia reciente para no bloquear

# ✅ NUEVO ENDPOINT: Registrar una nueva persona (con foto y estado de requisitoriado)
@app.post("/registrar_persona")
async def registrar_persona(
    file: UploadFile = File(...),
    nombre: str = Form(...),
    apellidos: str = Form(...),
    requisitoriado: bool = Form(...) # ¡CORREGIDO: Usamos 'requisitOriado' como la columna booleana!
):
    try:
        contents = await file.read()
        image = face_recognition.load_image_file(io.BytesIO(contents))
        
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            return JSONResponse(status_code=404, content={"message": "❌ No se detectó ningún rostro en la imagen."})

        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings:
            return JSONResponse(status_code=404, content={"message": "❌ No se pudo codificar el rostro de la imagen."})

        new_person_encoding = face_encodings[0].tolist() # Convertir a lista para guardar en JSON/Array en Supabase

        # 1. Subir la imagen a Supabase Storage
        file_name = f"{uuid.uuid4()}_{file.filename}" # Nombre único para el archivo
        try:
            # Usar el bucket 'rostros'
            response_storage = supabase.storage.from_("rostros").upload(file_name, contents, {"content-type": file.content_type})
            
            if response_storage.status_code != 200:
                print(f"Error al subir imagen a Supabase Storage: {response_storage.data}")
                return JSONResponse(status_code=500, content={"error": f"Error al subir imagen: {response_storage.data}"})
            
            # Obtener la URL pública de la imagen
            public_url_response = supabase.storage.from_("rostros").get_public_url(file_name)
            
            # Asegurarse de que la URL es un string y no un objeto con un 'data' o similar
            foto_url = public_url_response.data.get('publicUrl') if hasattr(public_url_response.data, 'get') else public_url_response.data
            
            if not foto_url:
                 # Fallback para versiones antiguas de Supabase client o si get_public_url no devuelve publicUrl
                supabase_url_base = supabase.url.replace(".supabase.co", ".storage.supabase.co/storage/v1/object/public/")
                foto_url = f"{supabase_url_base}rostros/{file_name}"

        except Exception as storage_e:
            print(f"Excepción al subir imagen o obtener URL: {storage_e}")
            return JSONResponse(status_code=500, content={"error": f"Error en Storage: {str(storage_e)}"})


        # 2. Insertar los datos de la persona en la tabla 'personas'
        response_db = supabase.table("personas").insert({
            "nombre": nombre,
            "apellidos": apellidos,
            "kp": new_person_encoding, # Guardamos el encoding
            "foto": foto_url, # Guardamos la URL de la foto
            "requisitoriado": requisitoriado # ¡CORREGIDO: Guardamos el estado de requisitoriado!
        }).execute()

        if response_db.data:
            return JSONResponse(status_code=200, content={"message": "✅ Persona registrada exitosamente.", "persona_id": response_db.data[0]["id"]})
        else:
            print(f"Error al insertar en DB: {response_db.data}")
            return JSONResponse(status_code=500, content={"error": f"Error al registrar persona en la base de datos: {response_db.data}"})

    except Exception as e:
        print(f"Error general en registrar_persona: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ Endpoint para reconocer un rostro (modificado para 'requisitOriado')
@app.post("/reconocer")
async def reconocer_rostro(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = face_recognition.load_image_file(io.BytesIO(contents))

        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            return JSONResponse(status_code=404, content={"message": "❌ No se detectó ningún rostro."})

        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings:
            return JSONResponse(status_code=404, content={"message": "❌ No se pudo codificar el rostro."})

        encoding_actual = face_encodings[0]

        # Seleccionamos también la columna 'requisitOriado'
        personas = supabase.table("personas").select("id, nombre, apellidos, kp, requisitOriado").execute() # ¡CORREGIDO!
        matches = []

        for persona in personas.data:
            kp = persona["kp"]
            if kp is None:
                continue # Saltar si el keypoint es nulo

            stored_encoding = np.array(kp)

            match = face_recognition.compare_faces([stored_encoding], encoding_actual, tolerance=0.5)
            if match[0]:
                # Verificar si ya se registró en la última hora
                if not ya_tiene_asistencia_reciente(persona["id"]):
                    ahora = datetime.now()
                    supabase.table("asistencias").insert({
                        "persona_id": persona["id"],
                        "fecha": ahora.date().isoformat(),
                        "hora": ahora.time().strftime("%H:%M:%S")
                    }).execute()

                matches.append({
                    "id": persona["id"],
                    "nombre": persona["nombre"],
                    "apellidos": persona["apellidos"],
                    "requisitOriado": persona.get("requisitOriado", False) # ¡CORREGIDO!
                })
                
                # ¡SIMULACIÓN DE ALERTA A LA POLICÍA!
                if persona.get("requisitOriado", False): # ¡CORREGIDO!
                    print(f"\n🚨🚨 ALERTA DE SEGURIDAD 🚨🚨")
                    print(f"¡PERSONA REQUISITORIADA DETECTADA!: {persona['nombre']} {persona['apellidos']} (ID: {persona['id']})")
                    print(f"Fecha: {datetime.now().isoformat()}\n")
                    # Aquí es donde integrarías un servicio real de SMS (ej. Twilio)
                    # para enviar un mensaje a la policía o a otro estudiante.

        if matches:
            return {"message": "✅ Rostro reconocido", "coincidencias": matches}
        else:
            return {"message": "❌ Rostro no reconocido"}

    except Exception as e:
        print(f"Error general en reconocer_rostro: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)