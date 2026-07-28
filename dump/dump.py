import argparse
import yaml

# Parse command line arguments.
parser = argparse.ArgumentParser(description='dump eval data.')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--base_dir', type=str, required=True)
parser.add_argument('--num_kpt', type=int, required=True)


def get_dumper(name):
    mod = __import__('dumper.{}'.format(name), fromlist=[''])
    return getattr(mod, name)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.Loader)
        config['rawdata_dir'] = args.base_dir + config['rawdata_dir']
        config['feature_dump_dir'] = args.base_dir + config['feature_dump_dir'] + f'_{args.num_kpt}'
        config['dataset_dump_dir'] = args.base_dir + config['dataset_dump_dir'] + f'_{args.num_kpt}'
        config['extractor']['num_kpt'] = args.num_kpt
        if config['data_name'] == 'megadepth1500':
            config['pairs'] = args.base_dir + config['pairs']

    dataset = get_dumper(config['data_name'])(config)

    dataset.get_seqs()
    dataset.format_dump_folder()
    if config['extractor']['extract']:
        dataset.dump_feature()
    dataset.format_dump_data()
