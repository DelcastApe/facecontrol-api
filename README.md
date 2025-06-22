# 👁️‍🗨️ facecontrol-api

Sistema completo de reconocimiento facial con control de asistencia, alertas por requisitoriados, generación de reportes y panel administrativo.

---

## 🚀 Tecnologías utilizadas

- **FastAPI** – Backend rápido y moderno
- **Supabase** – Base de datos, almacenamiento y autenticación
- **Face Recognition** – Modelo de extracción de vectores faciales (128D)
- **FPDF** – Generación de reportes en PDF
- **Twilio / SMTP** – Alertas por SMS y correo

---

## 🧠 Funcionalidades principales

✅ Registro de personas con su foto  
✅ Detección y comparación de rostros  
✅ Control de asistencia automático  
✅ Alertas por rostros marcados como “requisitoriados”  
✅ Dashboard con estadísticas  
✅ Generación de reportes en PDF  
✅ Entrenamiento adaptativo del modelo  
✅ Autenticación JWT para protección de rutas

---

## 📦 Instalación local

```bash
git clone https://github.com/DelcastApe/facecontrol-api.git
cd facecontrol-api
python -m venv venv
venv\Scripts\activate  # o source venv/bin/activate en Linux/Mac
pip install -r requirements.txt
