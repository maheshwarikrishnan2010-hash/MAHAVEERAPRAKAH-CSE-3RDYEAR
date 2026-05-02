def choose_format():

    print("\nSelect file format:")

    print("1 → CSV")
    print("2 → JSON")
    print("3 → Excel")

    choice = input("Enter option number: ")

    formats = {
        "1": "csv",
        "2": "json",
        "3": "xlsx"
    }

    return formats.get(choice, "csv")