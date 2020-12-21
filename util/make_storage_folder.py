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
    tracks = os.path.join(container_dir, 'tracks')
    for di in [train, valid, evalu]:
        try:
            os.mkdir(di)
            os.mkdir(os.path.join(di, 'data'))
            os.mkdir(os.path.join(di, 'labels'))
            os.mkdir(os.path.join(di, 'scatterer_positions'))
            os.mkdir(os.path.join(di, 'tracks'))
        except FileExistsError:
            pass
    try:
        os.mkdir(tracks)
        os.mkdir(os.path.join(tracks, 'training'))
        os.mkdir(os.path.join(tracks, 'training', 'data'))
        os.mkdir(os.path.join(tracks, 'training', 'labels'))
        os.mkdir(os.path.join(tracks, 'validation'))
        os.mkdir(os.path.join(tracks, 'validation', 'data'))
        os.mkdir(os.path.join(tracks, 'validation', 'labels'))
        os.mkdir(os.path.join(tracks, 'evaluation'))
        os.mkdir(os.path.join(tracks, 'evaluation', 'data'))
        os.mkdir(os.path.join(tracks, 'evaluation', 'labels'))
        
    except FileExistsError:
        pass
    return container_dir

def make_logs_folder(logs_dir: str = '') -> None:
    try:
        os.mkdir(logs_dir)
    except FileExistsError:
        pass

  
    