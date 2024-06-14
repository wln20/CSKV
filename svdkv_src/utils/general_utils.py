"""
General purpose utils
"""

def get_name(args):
    return f"density_k_{args.k_density}_v_{args.v_density}_quant_k_{args.k_bits}_v_{args.v_bits}_{'asvd_' if args.use_asvd else ''}{f'window_{args.q_window_size}_' if args.use_window else ''}"

def check_params(args):
    assert ((args.k_density < 1.0 and args.k_density > 0.0) and (args.v_density < 1.0 and args.v_density > 0.0)), \
        f"k_density and v_density must be within (0.0, 1.0), but got {args.k_density} and {args.v_density}"
         