from train.trainart import ArtClassifier

# Example usage
if __name__ == "__main__":
    art_classifier = ArtClassifier(
        "D:/RAHMA/Kuliah/Semester 6/Pengenalan Pola/pythonProject/dataset/Jenis Art/Pop Art",
        "D:/RAHMA/Kuliah/Semester 6/Pengenalan Pola/pythonProject/dataset/Jenis Art/Primitivism")
    art_classifier.train(100)
    art_classifier.evaluate(100)
    art_classifier.plot_confusion_matrix()
    art_classifier.compute_roc_auc()

