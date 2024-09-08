<?php

function dd()
{
    dump(...func_get_args());
    die;
}
function dump()
{
    foreach (func_get_args() as $arg) {
        var_dump($arg);
    }
}

function printResult(array $row, $array, array $result): void
{
    echo "Input: " . implode(', ', $row) . PHP_EOL;
    echo "Output: " . implode(', ', $array) . PHP_EOL;
    echo "Predict: " . implode(', ', array_map(fn($row) => round($row) . " ($row)", $result)) . PHP_EOL;
    echo PHP_EOL;
}