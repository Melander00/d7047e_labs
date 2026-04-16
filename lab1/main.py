from ann.ANN import exec_ann_amazon, exec_ann_yelp
from bert.BERT import exec_bert_amazon, exec_bert_yelp
from DistilBERT.DistilBERT import exec_distilbert_amazon, exec_distilbert_yelp
from gen_tensorboard import generate_tensorboard_files
from lstm.LSTM import exec_lstm_amazon, exec_lstm_yelp
from tqdm import tqdm
from transformers import logging

logging.set_verbosity_error()
logging.disable_progress_bar()


def main():
    yelp_entries = int(0.1 * 1e5) # 500K

    ann_learning_rate = 5e-5
    lstm_learning_rate = 1e-3
    bert_learning_rate = 5e-5

    ann_epochs = 2
    lstm_epochs = 2
    bert_epochs = 1

    ann_batch_size = 128
    lstm_batch_size = 128
    bert_batch_size = 16

    iteration_number = 1
    
    # print(f"Training models with {yelp_entries} yelp reviews. Iteration {iteration_number}")

    # exec_ann_amazon(use_25k_set=False, model_name="ann_amazon_1k", iteration_number=iteration_number, num_epochs=ann_epochs, batch_size=ann_batch_size, learning_rate=ann_learning_rate)
    # exec_ann_amazon(use_25k_set=True, model_name="ann_amazon_25k", iteration_number=iteration_number, num_epochs=ann_epochs, batch_size=ann_batch_size, learning_rate=ann_learning_rate)
    # exec_ann_yelp(entries=yelp_entries, model_name="ann_yelp", iteration_number=iteration_number, num_epochs=ann_epochs, batch_size=ann_batch_size, learning_rate=ann_learning_rate)

    # exec_lstm_amazon(use_25k_set=False, model_name="lstm_amazon_1k", iteration_number=iteration_number, num_epochs=lstm_epochs, batch_size=lstm_batch_size, learning_rate=lstm_learning_rate)
    # exec_lstm_amazon(use_25k_set=True, model_name="lstm_amazon_25k", iteration_number=iteration_number, num_epochs=lstm_epochs, batch_size=lstm_batch_size, learning_rate=lstm_learning_rate)
    # exec_lstm_yelp(entries=yelp_entries, model_name="lstm_yelp", iteration_number=iteration_number, num_epochs=lstm_epochs, batch_size=lstm_batch_size, learning_rate=lstm_learning_rate)

    # exec_bert_amazon(use_25k_set=False, model_name="bert_amazon_1k", iteration_number=iteration_number, num_epochs=bert_epochs, batch_size=bert_batch_size, learning_rate=bert_learning_rate)
    # exec_bert_amazon(use_25k_set=True, model_name="bert_amazon_25k", iteration_number=iteration_number, num_epochs=bert_epochs, batch_size=bert_batch_size, learning_rate=bert_learning_rate)
    # exec_bert_yelp(entries=yelp_entries, model_name="bert_yelp", iteration_number=iteration_number, num_epochs=bert_epochs, batch_size=bert_batch_size, learning_rate=bert_learning_rate)

    # exec_distilbert_amazon(use_25k_set=False, model_name="distilbert_amazon_1k", iteration_number=iteration_number, num_epochs=bert_epochs, batch_size=bert_batch_size, learning_rate=bert_learning_rate)
    # exec_distilbert_amazon(use_25k_set=True, model_name="distilbert_amazon_25k", iteration_number=iteration_number, num_epochs=bert_epochs, batch_size=bert_batch_size, learning_rate=bert_learning_rate)
    # exec_distilbert_yelp(entries=yelp_entries, model_name="distilbert_yelp", iteration_number=iteration_number, num_epochs=bert_epochs, batch_size=bert_batch_size, learning_rate=bert_learning_rate)

    # print("Finished training")

    model_names = [
        "ann_amazon_1k", 
        # "ann_amazon_25k", 
        # "ann_yelp",
        "lstm_amazon_1k", 
        "lstm_amazon_25k", 
        # "lstm_yelp",
        # "bert_amazon_1k", 
        # "bert_amazon_25k", 
        # "bert_yelp",
        # "distilbert_amazon_1k", 
        # "distilbert_amazon_25k", 
        # "distilbert_yelp",
    ]

    for name in tqdm(model_names, desc="Generating tensorboard files", leave=False):
        generate_tensorboard_files(name, iteration_number)
    
    print("Finished generating tensorboard files.")
    

    

if __name__ == "__main__":
    main()