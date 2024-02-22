from grouped_gemm import backend
import torch
import warnings

from sys import stderr


class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        assert torch.count_nonzero(batch_sizes) != 0, "Input batch_sizes should not be all zeros!"
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = backend.gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = backend.gmm(
                lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        return agrad, bgrad, None, None


def gmm(a, b, batch_sizes, trans_b=False):
    return GroupedGemm.apply(a, b, batch_sizes, trans_b)

def sinkhorn_kernel(cost, tol=0.0001):
    return backend.sinkhorn(cost, tol)

class PermuteMoE(torch.autograd.Function):
  
  workspace_fw=None
  workspace_bw=None
  dtype=None
  max_token_num=0

  @staticmethod
  def forward(ctx, 
              unpermuted_inputs: torch.Tensor,
              expert_for_rows: torch.Tensor,
              max_token_num: int):

    # Device check
    if unpermuted_inputs.is_cpu:
      raise RuntimeError("The input \"unpermuted_inputs\" of permute op is on the device: CPU!")
    if expert_for_rows.is_cpu:
        warnings.warn("The input \"expert_for_rows\" of permute op is on the device: CPU!")
        expert_for_rows = expert_for_rows.cuda()

    # Shape check
    if unpermuted_inputs.size(0) != expert_for_rows.size(0):
      raise RuntimeError(f"permute op input \"expert_for_rows\" shape mismatch! "
                         f"Expect {unpermuted_inputs.size(0)}, but got {expert_for_rows.size(0)}.")

    # Data type check
    if expert_for_rows.dtype != torch.int32:
        warnings.warn("Got {} dtype for \"expert_for_rows\", will be casted into torch.int32".format(expert_for_rows.dtype))
        expert_for_rows = expert_for_rows.to(torch.int32)

    # Contiguous check
    if not unpermuted_inputs.is_contiguous():
      print("[Warning] The input \"unpermuted_inputs\" of permute op is discontiguous!", file=stderr)
      unpermuted_inputs = unpermuted_inputs.contiguous()
    if not expert_for_rows.is_contiguous():
      print("[Warning] The input \"expert_for_rows\" of permute op is discontiguous!", file=stderr)
      expert_for_rows = expert_for_rows.contiguous()

    if PermuteMoE.max_token_num < max_token_num:
      # print("Permute op workspace reset!")
      PermuteMoE.max_token_num = max_token_num
      PermuteMoE.workspace_fw = []
      PermuteMoE.workspace_bw = []

    if PermuteMoE.max_token_num < unpermuted_inputs.size(0) or PermuteMoE.dtype != unpermuted_inputs.dtype:
      # print("Permute op workspace reset!")
      PermuteMoE.max_token_num = unpermuted_inputs.size(0)
      PermuteMoE.dtype = unpermuted_inputs.dtype
      PermuteMoE.workspace_fw = []
      PermuteMoE.workspace_bw = []

    permuted_inputs, source_row_to_dest_row, PermuteMoE.workspace_fw = backend.permute(
      unpermuted_inputs,
      expert_for_rows,
      PermuteMoE.workspace_fw,
      PermuteMoE.max_token_num)

    ctx.source_row_to_dest_row = source_row_to_dest_row

    return permuted_inputs, source_row_to_dest_row

  @staticmethod
  def backward(ctx, permuted_inputs_grad, _):
    if not permuted_inputs_grad.is_contiguous():
      permuted_inputs_grad = permuted_inputs_grad.contiguous()
    source_row_to_dest_row = ctx.source_row_to_dest_row

    original_output, PermuteMoE.workspace_bw = backend.unpermute(
      permuted_inputs_grad,
      source_row_to_dest_row,
      PermuteMoE.workspace_bw,
      PermuteMoE.max_token_num)

    return original_output, None, None

class UnpermuteMoE(torch.autograd.Function):

  workspace_fw=None
  workspace_bw=None
  dtype=None
  max_token_num=0
  
  @staticmethod
  def forward(ctx,
              permuted_inputs: torch.Tensor,
              expert_for_rows: torch.Tensor,
              source_row_to_dest_row: torch.Tensor,
              max_token_num: int):

    # Device check
    if permuted_inputs.is_cpu:
        raise RuntimeError("The input \"permuted_inputs\" of unpermute op is on the device: CPU!")
    if expert_for_rows.is_cpu:
        warnings.warn("The input \"expert_for_rows\" of unpermute op is on the device: CPU!")
        expert_for_rows = expert_for_rows.cuda()
    if source_row_to_dest_row.is_cpu:
        warnings.warn("The input \"source_row_to_dest_row\" of unpermute op is on the device: CPU!")
        source_row_to_dest_row = source_row_to_dest_row.cuda()

    # Shape check
    if permuted_inputs.size(0) != expert_for_rows.size(0):
      raise RuntimeError(f"unpermute op input \"expert_for_rows\" shape mismatch! "
                         f"Expect {permuted_inputs.size(0)}, but got {expert_for_rows.size(0)}.")

    # Data type check
    if expert_for_rows.dtype != torch.int32:
        warnings.warn("Got {} dtype for \"expert_for_rows\", will be casted into torch.int32".format(expert_for_rows.dtype))
        expert_for_rows = expert_for_rows.to(torch.int32)
    if source_row_to_dest_row.dtype != torch.int32:
        warnings.warn("The data type of the input \"source_row_to_dest_row\" of unpermute op is Int64! "
            "The recommended type is int32.")
        source_row_to_dest_row = source_row_to_dest_row.to(torch.int32)

    # Contiguous check
    if not permuted_inputs.is_contiguous():
        warnings.warn("The input \"permuted_inputs\" of unpermute op is discontiguous!")
        permuted_inputs = permuted_inputs.contiguous()
    if not expert_for_rows.is_contiguous():
        warnings.warn("The input \"expert_for_rows\" of unpermute op is discontiguous!")
        expert_for_rows = expert_for_rows.contiguous()
    if not source_row_to_dest_row.is_contiguous():
        warnings.warn("The input \"source_row_to_dest_row\" of unpermute op is discontiguous!")
        source_row_to_dest_row = source_row_to_dest_row.contiguous()

    if UnpermuteMoE.max_token_num < max_token_num:
      # print("Unpermute op workspace reset!")
      UnpermuteMoE.max_token_num = max_token_num
      UnpermuteMoE.workspace_fw = []
      UnpermuteMoE.workspace_bw = []

    if UnpermuteMoE.max_token_num < permuted_inputs.size(0) or UnpermuteMoE.dtype != permuted_inputs.dtype:
      # print("Unpermute op workspace reset!")
      UnpermuteMoE.max_token_num = permuted_inputs.size(0)
      UnpermuteMoE.dtype = permuted_inputs.dtype
      UnpermuteMoE.workspace_fw = []
      UnpermuteMoE.workspace_bw = []

    ctx.expert_for_rows = expert_for_rows

    original_output, UnpermuteMoE.workspace_bw = backend.unpermute(
      permuted_inputs,
      source_row_to_dest_row,
      UnpermuteMoE.workspace_bw,
      UnpermuteMoE.max_token_num)
    
    return original_output
  
  @staticmethod
  def backward(ctx, unpermuted_inputs_grad):
    if not unpermuted_inputs_grad.is_contiguous():
      unpermuted_inputs_grad = unpermuted_inputs_grad.contiguous()
    expert_for_rows = ctx.expert_for_rows

    permuted_inputs, _, UnpermuteMoE.workspace_fw = backend.permute(
      unpermuted_inputs_grad,
      expert_for_rows,
      UnpermuteMoE.workspace_fw,
      UnpermuteMoE.max_token_num)

    return permuted_inputs, None, None, None

def permute(unpermuted_inputs, expert_for_rows, max_token_num):
  return PermuteMoE.apply(unpermuted_inputs, expert_for_rows, max_token_num)

def unpermute(permuted_inputs, expert_for_rows, source_row_to_dest_row, max_token_num):
  return UnpermuteMoE.apply(permuted_inputs, expert_for_rows, source_row_to_dest_row, max_token_num)