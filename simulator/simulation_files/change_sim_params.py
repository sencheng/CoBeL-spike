import argparse
import json

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", type=str)
    parser.add_argument("master_rng_seed", type=int)

    args = parser.parse_args()

    with open('parameter_sets/current_parameter/sim_params.json', 'r+') as f:
        sim_params = json.load(f)

        sim_params["data_path"] = args.data_path
        sim_params["master_rng_seed"] = args.master_rng_seed

        f.seek(0)
        json.dump(sim_params, f, indent=2)
        f.truncate()

if __name__ == "__main__":
    main()
