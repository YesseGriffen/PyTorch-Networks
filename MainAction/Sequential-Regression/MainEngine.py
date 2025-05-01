
#---------------------------------------------
# Context
# 
# This situation is simply predicting of a sequence, attempting to predict the next value from a long-sequence of almost random float numbers
# The data was generated using an algorithm that gives very trendy data, and is extremely violate. With large spikes and a low average.





#---------------------------------------------
# Create parameters for model and data
def create_params():
    window, step, batch_size = 15, 5, 12
    feature, forecast = 1, 1
    input_dim, dim_feedforward, output_dim = feature, 192, forecast
    d_model, nhead, num_layers, dropout = 40, 5, 3, 0.2
    data_params = [window, feature, step, forecast, batch_size]
    model_params = [input_dim, dim_feedforward, output_dim, d_model, nhead, num_layers, dropout]

    return data_params, model_params

#---------------------------------------------
# *I have seen errors if using too many data points 10,000+, with certain parameters/models, so make sure everything works correctly if using a large dataset*
if __name__ == "__main__":
    import torch
    import all_data_py
    from ActionEngine import ActionMain
    #---------------------------------------------
    # Using the last 2000 to train and the last 100 to act as new data
    data = torch.from_numpy(all_data_py.data[-2000:-100]).float()
    new_data = torch.from_numpy(all_data_py.data[-100:]).float()
    data_params, model_params = create_params()

    engine = ActionMain(data, data_params, model_params)

    for i in range(len(new_data)):
        engine.ds.append(new_data[i])
        engine.overtime_fit(epochs=3, batch_range=3)
        engine.true_forecast()
        engine.end_event()
        


    engine.graph_results()
    engine.winner_winner_chicken_dinner()
    