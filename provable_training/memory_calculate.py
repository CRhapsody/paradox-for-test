from torch import cuda



def pp_cuda_mem(stamp: str = '') -> str:
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    if not cuda.is_available():
        return ''

    return '\n'.join([
        f'----- {stamp} -----',
        f'Allocated: {sizeof_fmt(cuda.memory_allocated())}',
        f'Max Allocated: {sizeof_fmt(cuda.max_memory_allocated())}',
        f'Cached: {sizeof_fmt(cuda.memory_reserved())}',
        f'Max Cached: {sizeof_fmt(cuda.max_memory_reserved())}',
        f'----- End of {stamp} -----'
    ])

if __name__ == '__main__':
    print(pp_cuda_mem('Before training'))