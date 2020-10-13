import os

def make_storage_folder(container_dir: str = '', name: str = '') -> None:
    try:
        if not container_dir:
            container_dir = 'storage'
        if name:
            container_dir = os.path.join(container_dir, name)
        else:
            pass
    except FileExistsError:
        container_dir = os.path.join(container_dir, name+'_0')
    try:
        os.mkdir(container_dir)
    except FileExistsError:
        pass
    train = os.path.join(container_dir, 'training')
    valid = os.path.join(container_dir, 'validation')
    evalu = os.path.join(container_dir, 'evaluation')
    for di in [train, valid, evalu]:
        try:
            os.mkdir(di)
            os.mkdir(os.path.join(di, 'data'))
            os.mkdir(os.path.join(di, 'labels'))
        except FileExistsError:
            pass
    return container_dir


    