from Core.data import Data
from Core.config import Column
from Core.model import StockPricePredictor

#data.py file calls

data = Data()
data.read('Data/AAKASH.csv')
data.check_null_values()
data.clean_data()
print(Column.OPEN.value)
data.print_head()
data.print_description()
data.normalize()
data.visualize(Column.CLOSE.value)

#model.py file calls

predictor = StockPricePredictor(data.dataframe, data.scaler)
sequence_length = 10
sequences, targets = predictor.create_sequences(sequence_length=sequence_length)
predictor.prepare_data(sequences, targets, sequence_length)
predictor.build_model(sequence_length)
history = predictor.train_model()
predictor.visualize_predictions()
predictor.save_model('PreTrainedModel/AAKASH_lstm_stock_model.pkl')
predictor.load_model('PreTrainedModel/AAKASH_lstm_stock_model.pkl')
results = predictor.run_predictions_and_evaluation()