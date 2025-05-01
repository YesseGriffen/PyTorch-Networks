import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ModelEngine import ModelOH
from DataEngine import TSD, TailS

#---------------------------------------------
# All the actions used for the model, such as training/fitting and predicting
class ActionMain:
    def __init__(self, data, data_params, model_params):
        self.data = data
        self.window, self.features, self.step, self.forecast, self.batch_size = data_params
        
        self.ds = TSD(series=data, data_params=data_params)
        self.moh = ModelOH(model_params=model_params, lr=1e-4)

        self.preds, self.trues = [], []
        
        self.init_fit(epochs=500)
    #---------------------------------------------
    # Initial fit set up any variables, etc. Only use this ONCE
    def init_fit(self, epochs: int):
        ts = TailS(dataset=self.ds, batch_size=self.batch_size)
        dl = DataLoader(dataset=self.ds, batch_sampler=ts)
        self.moh.fit(loader=dl, epochs=epochs)
    #---------------------------------------------
    # Large fit for warm-restarts, full-restarts, or anytime a larger fit is needed
    def large_fit(self, epochs: int):
        ts = TailS(dataset=self.ds, batch_size=self.batch_size)
        dl = DataLoader(dataset=self.ds, batch_sampler=ts)
        self.moh.fit(loader=dl, epochs=epochs)
    #---------------------------------------------
    # Overtime fit for smaller update training
    def overtime_fit(self, epochs: int, batch_range: int = 0, lr_factor: int = 0.1):
        ts = TailS(dataset=self.ds, batch_size=self.batch_size, batch_range=batch_range)
        dl = DataLoader(dataset=self.ds, batch_sampler=ts)
        self.moh.update_fit(loader=dl, epochs=epochs, lr_factor=lr_factor)
    #---------------------------------------------
    # Return last window for prediction and other uses *If using stateful model, will need to return batch_size similar to states or reinit states*
    def last_window(self):
        buf_list = list(self.ds.buffer)[-self.window:]
        Xw = torch.tensor(buf_list).unsqueeze(0).unsqueeze(-1).to(self.moh.device)
        return Xw
    #---------------------------------------------
    # Used to make a true forecast *Made for forecast_size = 1*
    def true_forecast(self):
        Xw = self.last_window()
        pred = self.moh.forecast(Xw)
        clean_prediction = round(pred.cpu().item() , 2)
        self.preds.append(clean_prediction)













    def end_event(self):
        last_value = list(self.ds.buffer)[-1]
        self.trues.append(last_value)

    def winner_winner_chicken_dinner(self):
        greater_than_count = 0
        main_count = 0
        for i in range(len(self.trues)):
            if self.trues[i] > self.preds[i]:
                main_count += self.preds[i] - 1
                greater_than_count += 1
            else:
                main_count -= 1
        print(f'Percentage --------- {(greater_than_count / len(self.trues)) * 100} --------- Gain 1x: {main_count}')
    def graph_results(self):
        

        plt.figure(figsize=(8, 4))
        plt.plot(self.trues, label="True")
        plt.plot(self.preds, label="Predicted", alpha=0.7)
        plt.xlabel("Sample Index")
        plt.ylabel("Target Value")
        plt.legend()
        plt.title("Predictions vs True Targets")
        plt.show()


