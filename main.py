from predict import predict_email
from train import train_model


def main():
    print("Training the spam detection model...")
    train_model()

    print("\nExample prediction:")
    print(predict_email("Congratulations! You have won a free prize. Claim now."))


if __name__ == "__main__":
    main()
