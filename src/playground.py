from src.predictions import get_predictions
import datetime

index = '^GSPC'

today = datetime.datetime.today().strftime('%Y-%m-%d')
predictions = get_predictions(index)
prediction = predictions[0]['date'].strftime('%Y-%m-%d')
print(today)
print(predictions)