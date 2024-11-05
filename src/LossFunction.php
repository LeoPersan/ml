<?php

namespace LeoPersan\ML;

class LossFunction
{
    /**
     * @param  float[][]  $output
     * @param  float[][]  $target
     * @return float[]
     */
    public function calculate(array $output, array $target): array
    {
        $error = [];
        foreach ($output as $i => $row) {
            foreach ($row as $j => $value) {
                $error[$j][] = $value - $target[$i][$j];
            }
        }
        foreach ($error as $i => $row) {
            $error[$i] = array_sum($row) / count($row);
        }

        return $error;
    }
}