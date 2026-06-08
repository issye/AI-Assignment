# product_categories.csv — Derived Lookup Table

## What this file is

`product_categories.csv` is a **derived lookup table** mapping each product (`StockCode`) in
the UCI Online Retail dataset to one of 8 system-defined categories.

It was **generated from** `Online Retail.xlsx` — it is NOT a modified version of that file.
`Online Retail.xlsx` is never written to or altered in any way.

---

## Columns

| Column | Type | Description |
|---|---|---|
| `StockCode` | string | Unique product identifier (matches Online Retail.xlsx) |
| `Description` | string | Product name in uppercase, whitespace stripped |
| `category` | string | Assigned system category (one of 8 values below) |

---

## How categories were assigned

Each product description was matched against keyword lists (uppercase). The first matching
category wins. If no keyword matches, the product defaults to `Home Decor` (the dominant
category in this dataset).

| Category | Keywords used |
|---|---|
| `Home Decor` | LANTERN, FRAME, CANDLE, VASE, MIRROR, SIGN, CLOCK, LIGHT, HOLDER, WALL |
| `Kitchen & Dining` | MUG, CUP, PLATE, BOWL, TEAPOT, JUG, KITCHEN, CAKE, SPOON, JAR |
| `Seasonal & Gifts` | CHRISTMAS, XMAS, EASTER, HALLOWEEN, VALENTINE, BIRTHDAY, GIFT, WRAP |
| `Toys & Games` | TOY, GAME, PUZZLE, DOLL, BEAR, PLAY, CHILDREN, KIDS |
| `Stationery & Craft` | PEN, CARD, NOTEBOOK, CRAFT, PAPER, STAMP, STICKER, TAPE |
| `Fashion & Accessories` | BAG, SCARF, JEWEL, NECKLACE, BRACELET, PURSE, UMBRELLA, WALLET |
| `Garden & Outdoor` | GARDEN, PLANT, OUTDOOR, WATERING, POT, BIRD, FLOWER |
| `Food & Confectionery` | FOOD, CHOCOLATE, SWEET, BISCUIT, JAM, HONEY, TEA, COFFEE |

---

## Coverage (3,665 products)

| Category | Products | Share |
|---|---|---|
| Home Decor | 2,064 | 56.3% |
| Fashion & Accessories | 322 | 8.8% |
| Kitchen & Dining | 316 | 8.6% |
| Seasonal & Gifts | 271 | 7.4% |
| Garden & Outdoor | 266 | 7.3% |
| Stationery & Craft | 243 | 6.6% |
| Toys & Games | 93 | 2.5% |
| Food & Confectionery | 90 | 2.5% |

Home Decor is dominant because it is also the default category. The keyword matching achieves
~44% explicit coverage; the remainder defaults to Home Decor, which is appropriate given the
dataset's UK gift/home decor retailer domain.

---

## How to regenerate

Run Section 1–2 of `notebooks/ml_model.ipynb` with `Kernel → Restart & Run All`.
The script reads `Online Retail.xlsx`, applies keyword matching, and overwrites this file.

---

## Source dataset

- **File:** `data/online+retail/Online Retail.xlsx`
- **Origin:** UCI Machine Learning Repository — Online Retail Dataset
- **Records:** 541,909 transactions (397,884 after cleaning)
- **Period:** Dec 2010 – Dec 2011, UK-based online gift retailer
- **Licence:** Public domain (UCI ML Repository)
