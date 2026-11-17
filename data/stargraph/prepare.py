import torch
import random
import os
import argparse


def general_tree_graph_maker(
    numOfPathsFromSource, lenOfEachPath, maxNodes, shuffle_edge_lists, reverse=False
):
    success = False
    while not success:
        nodes = list(range(maxNodes))
        random.shuffle(nodes)

        source = nodes.pop()
        existing_nodes = [source]
        edge_dict = {source: set()}
        # Populate edges
        for _ in range(numOfPathsFromSource * (lenOfEachPath - 1) ):
            item = nodes.pop()
            connected_to = existing_nodes[random.randint(0, len(existing_nodes) - 1)]
            existing_nodes.append(item)
            edge_dict[item] = set([connected_to])
            edge_dict[connected_to].add(item)
        # Yield all edges via BFS
        edgeList = []
        frontier = [source]
        parent = {source: None}
        degree_lookup = {0: set([source])}
        curr_deg = 0
        while len(frontier) > 0:
            next_frontier = []
            curr_deg += 1
            curr_degree_set = set()
            for item in frontier:
                for neighbor in edge_dict[item]:
                    # Visited already...
                    if neighbor in parent:
                        continue
                    curr_degree_set.add(neighbor)
                    parent[neighbor] = item
                    next_frontier.append(neighbor)
                    edgeList.append((item, neighbor))
            frontier = next_frontier
            if len(curr_degree_set) > 0:
                degree_lookup[curr_deg] = curr_degree_set
        # Select the goal and path - if the graph is unfortunately too shallow, try agian...
        if (lenOfEachPath - 1) in degree_lookup:
            valid_goals = list(degree_lookup[lenOfEachPath - 1])
            goal = valid_goals[random.randint(0, len(valid_goals) - 1)]
            trace_item = goal
            path = []
            for _ in range(lenOfEachPath):
                path.append(trace_item)
                trace_item = parent[trace_item]
            # Since we construct backwards, "reverse" in output = leave it as is, etc.
            if not reverse:
                path.reverse()
            if shuffle_edge_lists:
                edgeList = [(a,b) if random.randint(0,1) == 0 else (b,a) for (a,b) in edgeList]
                random.shuffle(edgeList)
            success = True
            
    return edgeList, path, source, goal
                
# if its a sink graph, just reverse the path
def star_or_sink_graph_maker(
    numOfPathsFromSource, lenOfEachPath, maxNodes, shuffle_edge_lists, reverse=False
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

    if shuffle_edge_lists:
        # edgeList = [(a,b) if random.randint(0,1) == 0 else (b,a) for (a,b) in edgeList]
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
    general_tree=False,
):
    os.makedirs(data_dir, exist_ok=True)
    file_name_comps = [f"graph_{numOfPathsFromSource}_{lenOfEachPath}_sample_{numOfSamples}"]
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
                )
            else:
                edgeList, path, source, goal = star_or_sink_graph_maker(
                    numOfPathsFromSource,
                    lenOfEachPath,
                    maxNodes,
                    shuffle_edge_lists,
                    reverse,
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
        general_tree=args.general_tree
    )


# python generate_graph.py --num_samples 500000 --num_paths 2 --path_length 5 --max_nodes 500 --data_dir "data/stargraph"

