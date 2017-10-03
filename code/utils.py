import os

def get_home_dir(repo_name='crypto_predict'):
    cwd = os.getcwd()
    cwd_list = cwd.split('/')
    repo_position = [i for i, s in enumerate(cwd_list) if s == repo_name]
    if len(repo_position) > 1:
        print("error!  more than one intance of repo name in path")
        return None

    home_dir = '/'.join(cwd_list[:repo_position[0] + 1]) + '/'
    return home_dir


def pull_s3():
    """
    Syncs all data from amazon S3.
    """


def push_s3():
    """
    Pushes up data to amazon S3.
    """
