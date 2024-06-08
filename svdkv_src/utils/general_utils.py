"""
General purpose utils
"""

def get_name(args):
    return f"compressed_dim_k_{args.k_compressed_dim}_v_{args.v_compressed_dim}_quant_k_{args.k_bits}_v_{args.v_bits}_{'asvd_' if args.use_asvd else ''}{f'window_{args.q_window_size}_' if args.use_window else ''}"