from dataset.loader import prepare_yelp_loaders


def main():
    loaders = prepare_yelp_loaders(entries=25000)
    

if __name__ == "__main__":
    main()