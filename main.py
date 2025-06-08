from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
from supabase_client import supabase
import io
from datetime import datetime, timedelta

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

# ✅ Nuevo endpoint para el dashboard
@app.get("/asistencias")
def get_asistencias():
    try:
        response = supabase.table("asistencias") \
            .select("persona_id, fecha, hora, personas(nombre, apellidos)") \
            .order("fecha", desc=True) \
            .order("hora", desc=True) \
            .limit(10) \
            .execute()
        
        # Formateamos la respuesta para devolver nombre completo + timestamp
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

# Verifica si ya registró asistencia hace menos de una hora
def ya_tiene_asistencia_reciente(persona_id: str) -> bool:
    ahora = datetime.now()
    fecha_actual = ahora.date().isoformat()
    hora_limite = (ahora - timedelta(hours=1)).time().isoformat()

    asistencias = supabase.table("asistencias") \
        .select("hora") \
        .eq("persona_id", persona_id) \
        .eq("fecha", fecha_actual) \
        .execute()

    for registro in asistencias.data:
        if registro["hora"] > hora_limite:
            return True
    return False

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

        personas = supabase.table("personas").select("id, nombre, apellidos, kp").execute()
        matches = []

        for persona in personas.data:
            kp = persona["kp"]
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
                    "apellidos": persona["apellidos"]
                })

        if matches:
            return {"message": "✅ Rostro reconocido", "coincidencias": matches}
        else:
            return {"message": "❌ Rostro no reconocido"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
