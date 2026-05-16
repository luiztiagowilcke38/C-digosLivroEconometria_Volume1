library(keras)
library(tensorflow)

# Dados de Inflacao sintetica mensalizada
set.seed(123)
n <- 240
inflacao <- cumsum(rnorm(n, 0.002, 0.003)) + 0.04

# Formatar janela deslizante
window_size <- 12
X_list <- list()
y_list <- c()
for(i in 1:(n-window_size)){
    X_list[[i]] <- inflacao[i:(i+window_size-1)]
    y_list[i]   <- inflacao[i+window_size]
}
X_arr <- array(unlist(X_list), dim=c(length(X_list), window_size, 1))
y_arr <- array(y_list, dim=c(length(y_list), 1))

# Modelo GRU (similar ao LSTM porem mais leve)
modelo_gru <- keras_model_sequential() |>
  layer_gru(units=64, return_sequences=TRUE, input_shape=c(window_size, 1)) |>
  layer_dropout(0.2) |>
  layer_gru(units=32, return_sequences=FALSE) |>
  layer_dense(units=1)

modelo_gru |> compile(optimizer=optimizer_adam(lr=0.001), loss='mse')

history <- modelo_gru |> fit(
    X_arr, y_arr,
    epochs = 100, batch_size = 32,
    validation_split = 0.2, verbose = 0)
cat(''Loss final:'', tail(history$metrics$val_loss, 1), ''\n'')
