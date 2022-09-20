import argparse
import json
import os
import time

import redis
import yaml

test_dir = os.path.dirname(__file__)
parser = argparse.ArgumentParser()
parser.add_argument('-d',
                    '--duration',
                    required=False,
                    type=int,
                    help="time (seconds) to wait before stopping the graph")
parser.add_argument("-g",
                    "--graph",
                    default=os.path.join(test_dir, 'testGraph.yaml'),
                    required=False,
                    help="path to graph file")
parser.add_argument('-f',
                    "--from-path",
                    default=False,
                    required=False,
                    action='store_true',
                    help="whether to send a path instead of a JSON when"
                    " starting a graph")
parser.add_argument('-c',
                    "--clean-up",
                    default=False,
                    required=False,
                    action='store_true',
                    help="whether to delete streams upon stopping the graph")
args = parser.parse_args()

with open(args.graph, 'r') as f:
    graph = yaml.safe_load(f)

r = redis.Redis()

_, start_streams = r.scan(0, _type='stream')

if args.from_path:
    print(f'Starting graph from {args.graph} as file path')
    r.xadd('supervisor_ipstream', {
        'commands': 'startGraph',
        'file': os.path.abspath(args.graph)
    })
else:
    print(f'Starting graph from {args.graph} as JSON')
    r.xadd('supervisor_ipstream', {
        'commands': 'startGraph',
        'graph': json.dumps(graph)
    })

if args.duration:
    print(f'Waiting {args.duration} seconds')
    time.sleep(args.duration)
else:
    input('Hit ENTER to stop graph...')

print('Stopping graph')
r.xadd('supervisor_ipstream', {'commands': 'stopGraph'})

if args.clean_up:
    # delete any streams created while the graph was running
    _, stop_streams = r.scan(0, _type='stream')

    new_streams = [
        stream for stream in stop_streams if stream not in start_streams
    ]

    i = 0
    while max([r.xlen(stream) for stream in new_streams]):
        for stream in new_streams:
            r.delete(stream)
        i += 1
    r.memory_purge()
    print(f'Deleted streams: {new_streams}')
