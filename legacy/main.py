import asyncio

import argparse
import logging

from cortex import Cortex
from sensors import Array

from sklearn import datasets

logging.basicConfig(level=logging.INFO)


def initialize_cortex():
    cortex = Cortex()
    array_sensor = Array(name='dummy', cortex=cortex)
    cortex.set_sensor(array_sensor)
    cortex.save('cortex.data')
    return cortex


def load_cortex():
    cortex = Cortex.load('cortex.data')
    return cortex


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--initialize",
        action="store_true",
        help="Initialize cortex data"
    )
    return parser.parse_args()


async def main():

    args = parse_arguments()

    if args.initialize:
        cortex = initialize_cortex()
    else:
        cortex = load_cortex()

    array_sensor = cortex.get_sensor('dummy')

    digits = datasets.load_digits()

    n1 = [
        digits.images[1].flatten(),
        digits.images[11].flatten(),
        digits.images[21].flatten(),
        digits.images[31].flatten(),
    ]

    n2 = [
        digits.images[2].flatten(),
        digits.images[12].flatten(),
        digits.images[22].flatten(),
        digits.images[32].flatten(),
    ]

    for _ in range(10):
        for array in n1:
            await array_sensor.send(array)

    # await array_sensor.send(n1[1])

    logging.info("[FIRE|QUEUE] start consuming firing queue")
    while not cortex.queue.empty():
        future = cortex.queue.get()
        await asyncio.ensure_future(future)

    for layer in cortex.children[0].children:
        logging.info("[LAYER] - %s - activated neurons: %s", layer, layer.activated_neurons)

    cortex.save('cortex.data')

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
