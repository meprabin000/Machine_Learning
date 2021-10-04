def file_stats( pathname ):   
    all_items = []
    sqrt = lambda x : x ** 0.5
    with open( pathname, "r") as f:
            all_items = ([(float(line)) for line in f])
            
    avg = sum(all_items) / len(all_items)    
    std = sqrt(sum([(item - avg)**2 for item in all_items]) / (len(all_items) - 1) )
    return avg, std
