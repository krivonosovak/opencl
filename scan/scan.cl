#define SWAP(a,b) {__local float * tmp=a; a=b; b=tmp;}

__kernel void scan_hillis_steele(__global float * input, 
                                __local float * a, __local float * b, 
                                int level, int array_size)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
 
    // level = block_size^k k - level of recursion
    int level_gind = level * (gid + 1) - 1; 
    level_gind = (level_gind > array_size && ((array_size /level + 1) * level - 1) == level_gind)
        ? array_size - 1
        : level_gind;
    
    if (level_gind < array_size) {
        a[lid] = b[lid] = input[level_gind];
    } else {
        a[lid] = b[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s = 1; s < block_size; s <<= 1)
    {
        if(lid > (s-1))
        {
            b[lid] = a[lid] + a[lid-s];
        }
        else
        {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }
    
    if(level_gind < array_size) {
        input[level_gind] = a[lid]; 
    }
}

__kernel void merge(__global float * input, __local float * a,
                        int level, int array_size)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    
    // level = block_size^k k - level of recursion
    int level_gind = level * (gid + 1) - 1;
    level_gind = (level_gind > array_size && level_gind == ((array_size / level  + 1) * level - 1))
        ? array_size - 1
        : level_gind;
     
    bool last_in_level_block = lid == (block_size - 1) && level_gind < array_size ||
                                level_gind == (array_size - 1);
    __local float prev_val;
    
    // read element before current level_block and elements of current level block
    int next_level = level * block_size;
    int prev_val_indx = (level_gind / next_level) * next_level - 1 ;
    
    if (last_in_level_block && prev_val_indx > 0) {
        prev_val = input[prev_val_indx];
    } else if (last_in_level_block && prev_val_indx < 0) {
        prev_val = 0;
    } else if (level_gind < array_size && !last_in_level_block) {
        a[lid] = input[level_gind];
    } else {
        a[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (!last_in_level_block && level_gind < array_size) {
        a[lid] += prev_val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
     if (!last_in_level_block && level_gind < array_size) {
        input[level_gind] = a[lid];
    }
    
}