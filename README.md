# Sea Turtle Re-ID (`seaTurtle_project`)

Bu proje SeaTurtleIDHeads veri seti üzerinde deniz kaplumbağası re-identification amacıyla geliştirilmiştir. Model, ResNet50 tabanlı embedding çıkarıcı olarak tasarlanmış ve Triplet Loss ile eğitilmiştir. Test aşamasında cosine similarity kullanılarak en benzer görüntüler bulunur ve Top-1 retrieval accuracy hesaplanır.

## Yapı

```
agents/
├── data_agent.py          # split → data_split/
├── preprocessing_agent.py # görüntü + embedding mimarisi
├── training_agent.py      # triplet eğitimi → models/turtle_reid_model.keras
├── testing_agent.py       # retrieval metrikleri (dosya yazmaz)
├── predict_agent.py       # tek görsel, top-5 kimlik
└── result_agent.py        # JSON çıktıları
```

Kaynak veri varsayılanı: proje klasörünün bir üst dizinindeki `dataset/` (veya ortam değişkeni `SEA_TURTLE_DATASET` ile özel yol).

## Çalıştırma sırası

1. Veri hazırlama:

```bash
python agents/data_agent.py
```

2. Eğitim:

```bash
python agents/training_agent.py
```

3. Test (konsol metrikleri):

```bash
python agents/testing_agent.py
```

4. Sonuçları dosyaya kaydetme (`test_results.json`, `test_predictions.json`, `test_predictions_accuracy.json`):

```bash
python agents/result_agent.py
```

5. Tek fotoğraf tahmini:

```bash
python agents/predict_agent.py --image "gorsel_yolu"
```

Tüm komutları proje kökünde (`seaTurtle_project/`) çalıştırın.

## Çıktılar

- `models/turtle_reid_model.keras` — L2-normalize 128 boyutlu embedding modeli
- `test_results.json` — özet metrikler
- `test_predictions.json` — görsel bazlı retrieval satırları
