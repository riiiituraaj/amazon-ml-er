from normalization import (
    normalize_name,
    normalize_address,
    extract_numbers,
)


examples = [
    {
        "name": "ABC Technologies Pvt. Ltd.",
        "address": "12, M.G. Road, Guwahati, Assam 781001",
    },
    {
        "name": "ABC Technologies Private Limited",
        "address": "12 MG Rd Guwahati Assam 781001",
    },
    {
        "name": "राम मार्केटिंग प्राइवेट लिमिटेड",
        "address": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
    },
]


for example in examples:

    print("=" * 70)

    print("Original name:")
    print(example["name"])

    print("\nNormalized name:")
    print(normalize_name(example["name"]))

    print("\nOriginal address:")
    print(example["address"])

    print("\nNormalized address:")
    print(normalize_address(example["address"]))

    print("\nNumbers:")
    print(extract_numbers(example["address"]))