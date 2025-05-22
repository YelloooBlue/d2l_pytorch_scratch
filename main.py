def main():
    print("Hello from pytorchstudy!")

    import torch
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")


if __name__ == "__main__":
    main()
