#! /bin/php
<?php

use LeoPersan\ML\ActivationFunction;
use LeoPersan\ML\Layer;
use LeoPersan\ML\NeuralNetwork;

require 'vendor/autoload.php';

$input = [[0, 0], [0, 1], [1, 0], [1, 1]];
$output = [[0], [1], [1], [0]];

$nn = new NeuralNetwork();
$nn->addLayer(new Layer(2, 4, new ActivationFunction()));
$nn->addLayer(new Layer(4, 1, new ActivationFunction()));
$nn->train($input, $output, 50000);

foreach ($input as $i => $row) {
    $result = $nn->predict($row);
    printResult($row, $output[$i], $result);
}

$nn->saveModel('model.json');

echo "Done!" . PHP_EOL;
