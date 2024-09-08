#! /bin/php
<?php

use LeoPersan\ML\NeuralNetwork;

require 'vendor/autoload.php';

$input = [[0, 0], [0, 1], [1, 0], [1, 1]];
$output = [[0], [1], [1], [0]];

$nn = new NeuralNetwork();
$nn->loadModel('model.json');

foreach ($input as $i => $row) {
    $result = $nn->predict($row);
    printResult($row, $output[$i], $result);
}

$nn->saveModel('model.json');

echo "Done!" . PHP_EOL;
