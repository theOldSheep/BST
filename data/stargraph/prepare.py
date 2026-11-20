import torch
import random
import os
import argparse


# A qcuik set union algorithm for generic tree graph connection
# Use this algorithm instead of "append the next node on existing cluster"
#   to prevent bias where expected degree of node is proportional to order of insertion
def set_lookup(lookup, i):
    if lookup[i] != i:
        asw = set_lookup(lookup, lookup[i])
        lookup[i] = asw
        return asw
    return i
def set_union(lookup, i, j):
    li, lj = set_lookup(lookup, i), set_lookup(lookup, j)
    if li != lj:
        lookup[li] = lj
        return True
    return False
def general_tree_graph_maker(
    numOfPathsFromSource,
    lenOfEachPath,
    maxNodes,
    shuffle_edge_lists,
    reverse=False,
    swap_edge_dir=False,
):
    success = False
    while not success:
        total_nodes = 1 + numOfPathsFromSource * (lenOfEachPath - 1)
        lookup = [i for i in range(0, total_nodes + 1)]
        num_comps = total_nodes
        edge_dict = {i: set() for i in range(1, total_nodes + 1)}
        # Grow the graph until becomes a tree
        edgeList = []
        while num_comps > 1:
            a,b = random.randint(1, total_nodes), random.randint(1, total_nodes)
            if set_lookup(lookup, a) == set_lookup(lookup, b):
                continue
            edgeList.append( (a,b) )
            edge_dict[a].add(b)
            edge_dict[b].add(a)
            set_union(lookup, a, b)
            num_comps -= 1
        # Try a few times to find a valid path
        for _ in range(10):
            source = random.randint(1, total_nodes)
            tgt = random.randint(1, total_nodes)
            # connect source & tgt via DFS
            stack = [source]
            parent = {source: None}
            while len(stack) > 0:
                curr = stack.pop()
                if curr == tgt:
                    break
                for neighbor in edge_dict[curr]:
                    # Visited already...
                    if neighbor in parent:
                        continue
                    parent[neighbor] = curr
                    stack.append(neighbor)
            # If the path is unfortunately too shallow, try sampling path agian...
            trace_item = tgt
            path = []
            while trace_item is not None:
                path.append(trace_item)
                trace_item = parent[trace_item]
            if len(path) >= lenOfEachPath:
                # Construct the full path in between
                # Clip to wanted path len
                padding = len(path) - lenOfEachPath
                prefix_pad = random.randint(0, padding)
                source, goal = path[prefix_pad], path[prefix_pad + lenOfEachPath - 1]
                path = path[prefix_pad : prefix_pad + lenOfEachPath]
                if reverse:
                    path.reverse()
                if swap_edge_dir:
                    edgeList = [(a,b) if random.randint(0,1) == 0 else (b,a) for (a,b) in edgeList]
                if shuffle_edge_lists:
                    random.shuffle(edgeList)
                success = True
                break
            
    return edgeList, path, source, goal
                
# if its a sink graph, just reverse the path
def star_or_sink_graph_maker(
    numOfPathsFromSource,
    lenOfEachPath,
    maxNodes,
    shuffle_edge_lists,
    reverse=False,
    swap_edge_dir=False
):
    nodes = list(range(maxNodes))
    random.shuffle(nodes)

    source = nodes.pop()

    edgeList = []
    path = [source]

    for p in range(numOfPathsFromSource):
        oldNode = source
        for i in range(lenOfEachPath - 1):
            newNode = nodes.pop()
            edgeList.append((oldNode, newNode))
            oldNode = newNode
            if p == 0:
                path.append(oldNode)
        if p == 0:
            goal = oldNode

    if swap_edge_dir:
        edgeList = [(a,b) if random.randint(0,1) == 0 else (b,a) for (a,b) in edgeList]
    if shuffle_edge_lists:
        random.shuffle(edgeList)

    if reverse:
        path = path[::-1]

    return edgeList, path, source, goal


def generate_and_save_star_or_sink_graph_data(
    numOfSamples,
    numOfPathsFromSource,
    lenOfEachPath,
    maxNodes,
    data_dir,
    shuffle_edge_lists=True,
    reverse=False,
    showLoadingBar=True,
    swap_edge_dir=False,
    general_tree=False,
):
    os.makedirs(data_dir, exist_ok=True)
    file_name_comps = [f"graph_{numOfPathsFromSource}_{lenOfEachPath}_sample_{numOfSamples}"]
    if swap_edge_dir:
        file_name_comps.append("_edgeswap")
    if general_tree:
        file_name_comps.append("_general_tree")
    with open(
        os.path.join(
            data_dir,
            f"{''.join(file_name_comps)}.txt",
        ),
        "w",
    ) as file:
        for x in range(numOfSamples):
            random.seed(x)
            if general_tree:
                edgeList, path, source, goal = general_tree_graph_maker(
                    numOfPathsFromSource,
                    lenOfEachPath,
                    maxNodes,
                    shuffle_edge_lists,
                    reverse,
                    swap_edge_dir,
                )
            else:
                edgeList, path, source, goal = star_or_sink_graph_maker(
                    numOfPathsFromSource,
                    lenOfEachPath,
                    maxNodes,
                    shuffle_edge_lists,
                    reverse,
                    swap_edge_dir,
                )
            file.write(
                "|".join([",".join([str(i) for i in x]) for x in edgeList])
                + f"/{source},{goal}={','.join([str(i) for i in path])}\n"
            )

            # loading bar
            if showLoadingBar:
                numberOfRectangles = int((x + 1) * 50 / numOfSamples)
                bar = "█" * numberOfRectangles + " " * (50 - numberOfRectangles)
                print(f"\r|{bar}| {(x+1)*100/numOfSamples:.1f}%", end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate star or sink graph data")
    parser.add_argument(
        "--num_samples", type=int, default=8000000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--num_paths", type=int, default=5, help="Number of paths from the source"
    )
    parser.add_argument(
        "--path_length", type=int, default=5, help="Length of each path"
    )
    parser.add_argument(
        "--max_nodes", type=int, default=50, help="Maximum number of nodes in the graph"
    )
    parser.add_argument(
        "--swap_edge_dir", type=bool, default=False, 
        help=   "Whether to generate the problem with 50 percent chance to flip edge direction, " \
                "I.e. some edges will be sent out in the form b,a instead of a,b"
    )
    parser.add_argument(
        "--general_tree", type=bool, default=False, 
        help=   "Whether generate a tree instead of stargraph, " \
                "with equivalent number of nodes to the stargraph configuration " \
                "and target path of exactly length of path_length"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/stargraph",
        help="Directory to save the generated data",
    )

    args = parser.parse_args()

    generate_and_save_star_or_sink_graph_data(
        numOfSamples=args.num_samples,
        numOfPathsFromSource=args.num_paths,
        lenOfEachPath=args.path_length,
        maxNodes=args.max_nodes,
        data_dir=args.data_dir,
        swap_edge_dir=args.swap_edge_dir,
        general_tree=args.general_tree
    )


# python generate_graph.py --num_samples 500000 --num_paths 2 --path_length 5 --max_nodes 500 --data_dir "data/stargraph"

