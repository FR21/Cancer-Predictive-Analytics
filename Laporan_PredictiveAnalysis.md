# Laporan Proyek Machine Learning Terapan - Felix Rafael
## Domain Proyek
Menurut **[Hanahan & Weinberg (2011)](https://doi.org/10.1016/j.cell.2011.02.013)**, Kanker merupakan suatu penyakit yang ditandai oleh _proliferasi_ sel abnormal secara tidak terkendali yang dapat menyerang jaringan di sekitarnya dan menyebar ke organ tubuh lain melalui proses _metastasis_. Secara biologis, kanker terjadi karena mutasi genetik yang memengaruhi jalur pengaturan siklus sel, _apoptosis_, dan mekanisme perbaikan DNA, yang menyebabkan sel kehilangan kemampuan untuk mengatur pertumbuhannya secara normal. Saat ini, kanker tetap menjadi tantangan kesehatan global yang signifikan, dengan dampak yang luas terhadap individu dan sistem kesehatan masyarakat. 

Menurut laporan **[International Agency for Research on Cancer (2020) ](https://doi.org/10.1016/j.cell.2011.02.013)**, terdapat hampir 20 juta kasus kanker baru dan sekitar 10 juta kematian akibat kanker di seluruh dunia pada tahun tersebut. Kanker paru-paru, payudara, dan kolorektal merupakan jenis yang paling umum, dengan kanker paru-paru menjadi penyebab utama kematian akibat kanker. Proyeksi yang ditunjukkan dari laporan **[ International Agency for Research on Cancer dan American Cancer Society  (2024) ](https://www.iarc.who.int/news-events/new-report-on-global-cancer-burden-in-2022-by-world-region-and-human-development-level/)** menunjukkan bahwa pada tahun 2050, jumlah kasus kanker baru tahunan dapat mencapai 35 juta, meningkat 77% dari angka tahun 2022.

Deteksi dini kanker sangat penting dalam meningkatkan hasil pengobatan dan kelangsungan hidup pasien. Namun, menurut penelitian yang dilakukan oleh **[Crosby et al. (2022) ](https://www.science.org/doi/10.1126/science.aay9040)** sekitar 50% kasus kanker didiagnosis pada stadium lanjut, yang secara signifikan mengurangi efektivitas pengobatan dan peluang kesembuhan. Studi oleh **[Cancer Research UK (2023) ](https://www.cancerresearchuk.org/about-cancer/spot-cancer-early/why-is-early-diagnosis-important)** menekankan bahwa diagnosis kanker pada tahap awal meningkatkan kemungkinan pengobatan yang berhasil dan kelangsungan hidup pasien. Oleh karena itu, strategi untuk meningkatkan deteksi dini sangat penting dalam upaya mengurangi beban kanker secara global.

Dalam konteks ini, pendekatan berbasis teknologi, khususnya machine learning (ML), menawarkan potensi besar dalam meningkatkan deteksi dan prediksi keparahan kanker. ML dapat menganalisis data klinis dan lingkungan pasien untuk mengidentifikasi pola yang mungkin tidak terlihat oleh metode konvensional. Studi oleh **[Zhou dan Rhrissorrakrai (2024)](https://doi.org/10.48550/arXiv.2410.22387)** menunjukkan bahwa ML dapat digunakan untuk menemukan biomarker multi-omik yang berkaitan dengan keparahan kanker prostat, yang dapat membantu dalam penilaian dan pengobatan pasien .

Proyek ini bertujuan untuk mengembangkan model prediksi keparahan kanker menggunakan beberapa algoritma Machine Learning. Dengan memanfaatkan data dari berbagai faktor genetik dan lingkungan, model diharapkan dapat memberikan prediksi yang akurat mengenai tingkat keparahan kanker pada pasien. Implementasi model dapat membantu dalam pengambilan keputusan klinis, perencanaan pengobatan, dan pada akhirnya, meningkatkan hasil kesehatan pasien secara keseluruhan.

## Business Understanding
### Problem Statements
- Tingkat keparahan kanker yang bervariasi antar pasien membuat diagnosis dan penanganan menjadi kompleks, ditambah kurangnya alat bantu berbasis kecerdasan buatan yang dapat memberikan prediksi tingkat keparahan kanker  kepada tenaga medis atau pasien.
- Belum adanya sistem prediksi terintegrasi yang menggabungkan faktor genetik, gaya hidup, dan lingkungan secara bersamaan dalam memperkirakan tingkat keparahan kanker. 
- Model baseline sederhana yang belum mampu memberikan akurasi prediksi yang tinggi dalam konteks regresi medis. 

### Goals
- Mengembangkan model Machine Learning berbasis regresi untuk memprediksi tingkat keparahan kanker dengan akurasi tinggi.
- Mengintegrasikan berbagai fitur dari data genetik, gaya hidup, dan lingkungan guna meningkatkan akurasi dan generalisasi model.
- Melakukan eksperimen dengan beberapa algoritma tingkat lanjut untuk menemukan model terbaik dalam memprediksi skor keparahan kanker, serta memvisualisasikan hasil evaluasi model untuk analisis residual yang lebih dalam.

### Solution Statement
