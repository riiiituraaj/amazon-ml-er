from normalization import (
    normalize_name_v2,
    normalize_name_token_sorted,
)


examples = [
    "Desert Society Inc",
    "Society Desert Inc",
    "INCORPORATED DESERT SOCIETY",
    "Sarasva India Limited",
    "Limited Sarasva India",
    "Clairvoyant Récord Private Limited",
    "Clairvoyant Record Private Ltd",
    "Rapid Transit Laboratories Cáre",
    "Rapid Transit Laboratories Care",
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "సుప్రీమ్ ఐటి ప్రైవేట్ లిమిటెడ్",
]


for name in examples:

    print("=" * 70)

    print("Original:")
    print(name)

    print("\nV2 normalized:")
    print(normalize_name_v2(name))

    print("\nToken sorted:")
    print(normalize_name_token_sorted(name))