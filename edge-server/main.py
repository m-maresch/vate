import argparse
import asyncio

from edge_server import EdgeServer
from services import get_predictor


def main(ipc: bool, jetson: bool):
    print(f"Got: ipc={ipc}, jetson={jetson}")

    predictor = get_predictor(jetson)
    edge_server = EdgeServer(predictor)
    edge_server.listen(ipc)

    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(edge_server.handle_requests())

    pending = asyncio.Task.all_tasks()
    event_loop.run_until_complete(asyncio.gather(*pending))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipc', action='store_true')
    parser.add_argument('--jetson', action='store_true')
    parser.add_argument('--no-jetson', dest='jetson', action='store_false')
    parser.set_defaults(ipc=False, jetson=True)
    args = parser.parse_args()

    main(args.ipc, args.jetson)
