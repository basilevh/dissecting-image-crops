'''
Image crop detection by absolute patch localization.
File management including checkpoints and logging.
Basile Van Hoorick, Fall 2020.
'''

# Library imports.
import os


def _find_latest_dir(inside_dir, must_contain=''):
    all_dirs = os.listdir(inside_dir)
    all_dirs = [os.path.join(inside_dir, dn) for dn in all_dirs if must_contain in dn]
    all_dirs = [dp for dp in all_dirs if os.path.isdir(dp)]
    all_dirs.sort(key=lambda dp: os.path.getmtime(dp))
    if len(all_dirs) == 0 and len(must_contain) != 0:
        raise RuntimeError('Failed to find configuration matching: ' + must_contain)
    return all_dirs[-1]


def _find_latest_checkpoint_file(inside_dir, must_contain=''):
    all_files = os.listdir(inside_dir)
    all_files = [os.path.join(inside_dir, fn) for fn in all_files
                 if fn.endswith('.pt') and must_contain in fn]
    all_files = [fp for fp in all_files if
                 os.path.isfile(fp)]
    all_files.sort(key=lambda fp: os.path.getmtime(fp))
    if len(all_files) == 0:
        return ''
    else:
        return all_files[-1]


def get_dirs_from_resume(checkpoint_dir, log_dir, image_dir, resume_path, model_tag):
    '''
    Parses given information to get full checkpoint, logging, and resume paths.
    Example:
    checkpoint_dir = '/path/to/checkpoints_h2'
    log_dir = '/path/to/logs_h2'
    image_dir = '/path/to/images_h2'
    resume_path = ''
    model_tag = '2020_10_20_h2'
    '''
    if resume_path == 'latest' or resume_path == 'last':
        # Find most recent global checkpoint.
        print('Finding latest checkpoint...')
        checkpoint_dir = _find_latest_dir(checkpoint_dir)

    elif resume_path == 'latest_same_conf' or resume_path == 'last_same_conf':
        # Find most recent global checkpoint with the same parameters.
        print('Finding latest checkpoint with the exact same parameters...')
        must_contain = model_tag[model_tag.index('_')+1:]  # Omit date.
        checkpoint_dir = _find_latest_dir(checkpoint_dir, must_contain=must_contain)

    elif os.path.isdir(resume_path):
        # Find most recent checkpoint within given run folder.
        print('Finding latest checkpoint within specified directory...')
        checkpoint_dir = resume_path

    elif len(resume_path) > 10 and os.path.isdir(os.path.join(checkpoint_dir, resume_path)):
        # Find most recent checkpoint within given run folder (by relative path).
        print('Finding latest checkpoint within specified directory...')
        resume_path = os.path.join(checkpoint_dir, resume_path)
        checkpoint_dir = resume_path

    elif os.path.isfile(resume_path):
        # Exact checkpoint file is specified.
        checkpoint_dir = os.path.dirname(resume_path)

    elif len(resume_path) > 10 and os.path.isfile(os.path.join(checkpoint_dir, resume_path)):
        # Exact checkpoint file is specified (by relative path).
        resume_path = os.path.join(checkpoint_dir, resume_path)
        checkpoint_dir = os.path.dirname(resume_path)

    else:
        # Do not rely on resume path.
        checkpoint_dir = os.path.join(checkpoint_dir, model_tag)

    # At this point: checkpoint_dir is one level deeper than the initial argument.

    if not os.path.isfile(resume_path):
        # Only the particular run directory is specified.
        model_tag = os.path.basename(checkpoint_dir)
        # Could remain unchanged and/or be empty if starting new run.
        if os.path.isdir(checkpoint_dir):
            resume_path = _find_latest_checkpoint_file(checkpoint_dir)

    if os.path.isdir(checkpoint_dir) and not(os.path.isfile(resume_path)):
        # Assume resume within existing checkpoint directory.
        print('===> WARNING: ' + checkpoint_dir + ' already exists, forcing resume!')
        resume_path = checkpoint_dir
        return get_dirs_from_resume(None, log_dir, image_dir, resume_path, model_tag)

    else:
        # Define actual (potentially new) subdirectories.
        log_dir = os.path.join(log_dir, model_tag)
        image_dir = os.path.join(image_dir, model_tag)

        return checkpoint_dir, log_dir, image_dir, resume_path, model_tag


def get_epoch_from_path(checkpoint_path):
    '''
    Returns the 0-based epoch as integer from something like /path/to/epoch7_train.pt.
    '''
    last_epoch = checkpoint_path.split('epoch')[1]
    last_epoch = last_epoch.split('_')[0]
    last_epoch = int(last_epoch)
    return last_epoch


def find_checkpoint_to_test(model_path):
    '''
    Ensures that the model path points to the precise file to test.

    Args:
        model_path: Exact file or containing checkpoint directory.

    Returns:
        model_path that is either the same, or more specific.
    '''
    if os.path.isdir(model_path):
        # Find 'best' if it exists, most recent otherwise.
        candidate = _find_latest_checkpoint_file(model_path, must_contain='best')
        if len(candidate) == 0:
            candidate = _find_latest_checkpoint_file(model_path)
            print('Found most recent checkpoint:', candidate)
        else:
            print('Found best checkpoint:', candidate)
        model_path = candidate
    assert(os.path.isfile(model_path))
    return model_path
