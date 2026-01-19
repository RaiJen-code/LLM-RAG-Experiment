# 📁 Knowledge Base Setup Guide

## 🎯 Direktori Structure yang Benar

Setelah Anda transfer semua file ke Jetson, struktur direktori akan seperti ini:

```
~/voice_assistant_research/
├── enhanced_assistant_v2.py          # Main program (sudah ada)
├── reference_answers.json            # Ground truth answers (sudah ada)
├── voice_testing_session.ipynb       # Testing notebook (sudah ada)
├── analysis_notebook.ipynb           # Analysis notebook (akan dibuat)
├── README_SETUP.md                   # File ini
│
├── knowledge_base/                   # ← FOLDER UNTUK KNOWLEDGE BASE
│   ├── default_knowledge/            # ← Untuk default knowledge (akan dibuat nanti)
│   │   └── (empty for now)           
│   │
│   └── user_documents/               # ← FOLDER UNTUK DOKUMEN ANDA ⭐
│       ├── (letakkan PDF Anda di sini)
│       ├── (letakkan TXT files di sini)
│       └── (sistem akan auto-load semua file)
│
├── assets/                           # Audio files (dari tutorial)
│   ├── bip.wav
│   └── bip2.wav
│
├── experiment_results/               # Results akan tersimpan di sini (auto-created)
│   └── (empty - akan diisi saat testing)
│
└── logs/                             # Logs (auto-created)
    └── (empty - akan diisi otomatis)
```

---

## 📥 **CARA MENAMBAHKAN DOKUMEN KNOWLEDGE BASE**

### **Langkah 1: Transfer File ke Jetson**

```bash
# Di Jetson, pastikan Anda di project directory
cd ~/voice_assistant_research

# Verifikasi folder sudah ada (auto-created by program)
ls -la knowledge_base/user_documents/
```

### **Langkah 2: Copy Dokumen Mekatronika Anda**

```bash
# Dari komputer lain via scp:
scp /path/to/your/dokumen*.pdf jetson@<jetson-ip>:~/voice_assistant_research/knowledge_base/user_documents/

# Atau jika file sudah di Jetson (dari USB drive, Downloads, etc):
cp ~/Downloads/dokumen_arduino.pdf ~/voice_assistant_research/knowledge_base/user_documents/
cp ~/Documents/sensor_guide.pdf ~/voice_assistant_research/knowledge_base/user_documents/
```

### **Langkah 3: Verifikasi File Sudah Ada**

```bash
cd ~/voice_assistant_research/knowledge_base/user_documents
ls -lh  # Lihat semua file

# Output expected:
# -rw-r--r-- 1 user user 2.3M Jan 16 10:00 arduino_basics.pdf
# -rw-r--r-- 1 user user 1.5M Jan 16 10:01 sensor_guide.pdf
# -rw-r--r-- 1 user user 800K Jan 16 10:02 electronics.txt
```

---

## ✅ **FILE FORMAT YANG DIDUKUNG**

### **PDF Files** (Recommended) ✅
- ✅ PDF dengan text (textbooks, documentation, articles)
- ✅ PDF dengan gambar + caption (gambar akan di-skip, text diambil)
- ⚠️ PDF hasil scan (image-only) - text tidak akan terekstrak
- **Contoh nama file yang bagus:**
  - `arduino_programming_guide.pdf`
  - `sensor_datasheet_collection.pdf`
  - `electronics_fundamentals.pdf`

### **Text Files** (.txt) ✅
- ✅ Plain text files
- ✅ UTF-8 encoding
- **Contoh:**
  - `mechatronics_notes.txt`
  - `iot_protocols_summary.txt`

### **JSON Files** (.json) ✅
- ✅ List of facts/documents
- **Format:**
  ```json
  {
    "documents": [
      "First fact or document...",
      "Second fact or document...",
      "Third fact or document..."
    ]
  }
  ```

---

## 🧪 **TEST KNOWLEDGE BASE LOADING**

Setelah menambahkan dokumen, test apakah berhasil di-load:

```bash
cd ~/voice_assistant_research

# Test loading
python3 enhanced_assistant_v2.py test
```

**Expected output:**
```
============================================================
Initializing Models...
============================================================
✓ Embedding model loaded: all-MiniLM-L6-v2
✓ Whisper model loaded: tiny
Loading default knowledge from: /home/.../default_knowledge
Loading user documents from: /home/.../user_documents
Successfully loaded PDF: arduino_basics.pdf
Successfully loaded PDF: sensor_guide.pdf
Successfully loaded text file: notes.txt
Total files loaded: 3

Knowledge Base Statistics:
  Total chunks: 287
  Total sources: 3
    - arduino_basics.pdf: 145 chunks
    - sensor_guide.pdf: 98 chunks
    - notes.txt: 44 chunks
```

---

## 📊 **REKOMENDASI DOKUMEN**

### **Prioritas Tinggi** (sangat penting untuk knowledge base yang bagus):

1. **Arduino Documentation** (5-20 halaman)
   - Arduino Uno specifications
   - Pin descriptions
   - Programming basics
   - Common libraries

2. **Sensor Datasheets/Guides** (3-10 halaman per sensor)
   - DHT11/DHT22 (temperature & humidity)
   - HC-SR04 (ultrasonic distance)
   - PIR motion sensor
   - Common specifications dan wiring

3. **Electronic Components Guide** (10-30 halaman)
   - Resistors (types, color codes)
   - Capacitors (types, applications)
   - LEDs (specifications, current limiting)
   - Transistors (NPN/PNP basics)
   - Diodes

4. **Communication Protocols** (5-15 halaman)
   - UART/Serial
   - I2C
   - SPI
   - Basic explanations dan comparisons

5. **Microcontroller Basics** (10-20 halaman)
   - Arduino vs ESP32
   - Digital vs Analog pins
   - PWM explanation
   - Interrupts vs Polling
   - ADC/DAC basics

### **Prioritas Sedang** (nice to have):

6. IoT Protocols (MQTT, HTTP)
7. Voltage regulators (linear vs switching)
8. Power management
9. Breadboard dan circuit basics
10. Troubleshooting guides

### **Format Ideal:**

- **Total**: 5-10 PDF files
- **Total pages**: 50-200 halaman
- **Language**: English (lebih konsisten dengan pertanyaan test)
- **Content**: Technical documentation, tutorials, atau textbooks

---

## ⚙️ **ADVANCED: Build Knowledge Base Index**

Jika Anda ingin pre-build index (opsional, untuk faster startup):

```bash
cd ~/voice_assistant_research

# Build index dari semua documents
python3 -c "
import enhanced_assistant_v2 as ea
stats = ea.kb.get_stats()
print(f'Knowledge base ready: {stats[\"total_chunks\"]} chunks')
"
```

---

## ❓ **TROUBLESHOOTING**

### **Problem: "No knowledge base files found"**

```bash
# Check if folders exist
ls -la knowledge_base/
ls -la knowledge_base/user_documents/

# If folders missing, create them
mkdir -p knowledge_base/user_documents
mkdir -p knowledge_base/default_knowledge
```

### **Problem: "PDF tidak terekstrak"**

PDF mungkin hasil scan (image-only). Solusi:
- Cari PDF yang bisa di-select text-nya (bukan gambar)
- Atau convert ke text manual
- Atau skip PDF tersebut

### **Problem: "Text encoding error"**

```bash
# Check file encoding
file -bi dokumen.txt

# Jika bukan UTF-8, convert:
iconv -f ISO-8859-1 -t UTF-8 dokumen.txt > dokumen_utf8.txt
```

---

## 📝 **CHECKLIST**

Sebelum mulai testing, pastikan:

- [ ] Folder `knowledge_base/user_documents/` sudah ada
- [ ] Minimal 3-5 PDF dokumen mekatronika sudah di-copy
- [ ] File format: PDF atau TXT
- [ ] Total ukuran: reasonable (< 50MB total)
- [ ] Test load dengan `python3 enhanced_assistant_v2.py test`
- [ ] Lihat output "Total chunks: XXX" (idealnya > 100 chunks)

---

## 🎓 **TIPS UNTUK KNOWLEDGE BASE YANG BAIK**

### ✅ **DO:**
- Gunakan dokumentasi resmi (Arduino.cc, manufacturer datasheets)
- Pilih dokumen yang jelas dan well-structured
- Fokus ke topik yang relevan dengan test questions
- Mix berbagai topik (sensors, components, protocols)

### ❌ **DON'T:**
- Jangan gunakan PDF yang kebanyakan gambar diagram tanpa text
- Jangan gunakan dokumen yang terlalu advanced/niche
- Jangan duplicate content (1 topik cukup 1 dokumen)
- Jangan terlalu banyak (>20 files) - quality > quantity

---

## 🚀 **NEXT STEPS**

Setelah knowledge base ready:

1. ✅ Verify dengan quick test
2. ✅ Open `voice_testing_session.ipynb`
3. ✅ Run testing session (20-30 menit)
4. ✅ Analyze results
5. ✅ Write journal paper! 📄

---

**Good luck! 🎯**
