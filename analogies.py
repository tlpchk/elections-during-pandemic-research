import fasttext as ft

if __name__ == '__main__':
    model = ft.load_model("model/election_model.bin")
    model.get_analogies("opozycja", "Kidawa", "PiS")
