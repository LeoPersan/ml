<?php

namespace LeoPersan\ML;

use JetBrains\PhpStorm\NoReturn;

function create_model(array $input, array $output): array
{
    $args   = is_array($input[0]) ? count($input[0]) : 1;

    return [
        'weights' => array_fill(0, $args, 1),
        'bias'    => 1,
    ];
}

function train_model(
    array $model,
    array $input,
    array $result,
    float $learning_rate = 0.001,
    int   $epochs = 100000,
    int   $batch_size = 10
): array {
    $input          = array_map(fn($x) => is_array($x) ? $x : [$x], $input);
    $order = range(1, count($input));
    shuffle($order);
    array_multisort($order, $input, $result);
    $batches        = array_chunk($input, $batch_size);
    $result_batches = array_chunk($result, $batch_size);
    for ($epoch = 0; $epoch < $epochs; $epoch ++) {
        foreach ($batches as $b_key => $batch) {
            $predicted = predict_batch($model, $batch);
            $model     = update_weights($model, $batch, $predicted, $result_batches[$b_key], $learning_rate);
            // dump($model);
        }
        if ($epoch % 100 == 0) {
            $mse = mean_squared_error($predicted, $result);
            echo "Epoch: " . $epoch . " Mean Squared Error: " . $mse . "\n";
            echo "Weights: " . implode(", ", $model['weights']) . " Bias: " . $model['bias'] . "\n";
        }
    }

    return $model;
}

function update_weights(array $model, array $batch, array $predicted, array $result, float $learning_rate): array
{
    $gradient = 0;
    foreach ($predicted as $key => $value) {
        $gradient += ($value - $result[$key]) / $result[$key];
    }
    $gradient /= count($predicted);
    foreach ($model['weights'] as $key => $weight) {
        $model['weights'][$key] -= $learning_rate * $gradient * $batch[0][$key];
        // $model['weights'][$key] -= $learning_rate * $gradient * $batch[0][$key];
    }
    $model['bias'] -= $learning_rate * $gradient * $model['bias'];

    return $model;
}

function predict_batch(array $model, mixed $batch): array
{
    $batch     = array_map(fn($x) => is_array($x) ? $x : [$x], $batch);
    $predicted = [];
    foreach ($batch as $input) {
        $predicted[] = predict($model, $input);
    }

    return $predicted;
}

function predict(array $model, array $input): float
{
    $output = 0;
    foreach ($input as $key => $value) {
        $output     += $value * $model['weights'][$key];
    }
    $output += $model['bias'];

    return $output;
}

function accuracy(array $predicted, array $result): float
{
    $correct = 0;
    foreach ($predicted as $key => $value) {
        if ($value == $result) {
            $correct ++;
        }
    }

    return $correct / count($predicted);
}

function mean_squared_error(array $predicted, array $result): float
{
    $error = 0;
    foreach ($predicted as $key => $value) {
        $error += ($value - $result[$key]) ** 2;
    }

    return $error / count($predicted);
}

#[NoReturn] function dd(): void
{
    dump(...func_get_args());
    die;
}

function dump(): void
{
    array_map(fn($x) => var_dump($x), func_get_args());
}