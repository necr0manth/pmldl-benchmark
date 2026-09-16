<#
.SYNOPSIS
    Launch Qwen3.5-4B LoRA / QLoRA fine-tuning with the optimal few-shot prompt.

.DESCRIPTION
    Trains Qwen3.5-4B using PEFT/LoRA and 4-bit quantization on the 110-photo museum dataset.
    Uses prompts/people-fewshot.txt (balanced attentive/inattentive exemplars) for check_people
    and preserves structured JSON tool calling without catastrophic forgetting.

.EXAMPLE
    # Test dataset, templating, and tokenization without training:
    .\scripts\run_finetune.ps1 -DryRun

.EXAMPLE
    # Run multi-task fine-tuning across all methods:
    .\scripts\run_finetune.ps1 -Epochs 1

.EXAMPLE
    # Run fine-tuning and merge adapter weights:
    .\scripts\run_finetune.ps1 -Epochs 1 -MergeAndSave
#>

param(
    [string]$ModelId = "Qwen/Qwen3.5-4B",
    [string]$TrainDataset = "datasets/train/cases.jsonl",
    [string]$OutputDir = "outputs/qwen3.5-4b-fewshot-lora",
    [string]$PeoplePromptPath = "prompts/people-fewshot.txt",
    [string]$DecisionPromptPath = "prompts/decision-v2.txt",
    [string]$DescriptionPromptPath = "prompts/description-system.txt",
    [string]$MethodFilter = "all",
    [string]$Quantization = "4bit",
    [int]$Epochs = 3,
    [int]$BatchSize = 1,
    [int]$GradAccum = 4,
    [string]$LearningRate = "2e-4",
    [int]$LoraR = 16,
    [int]$LoraAlpha = 32,
    [switch]$MergeAndSave,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "=== Qwen3.5-4B Few-Shot LoRA Fine-Tuning Launcher ===" -ForegroundColor Cyan
Write-Host "Model ID:            $ModelId"
Write-Host "Dataset:             $TrainDataset"
Write-Host "Method Filter:       $MethodFilter"
Write-Host "People Prompt:       $PeoplePromptPath (optimal few-shot exemplars)"
Write-Host "Quantization:        $Quantization"
Write-Host "Epochs:              $Epochs"
Write-Host "Batch Size:          $BatchSize (grad accum: $GradAccum, effective batch: $($BatchSize * $GradAccum))"
Write-Host "Output Directory:    $OutputDir"
Write-Host ""

# 1. Check Python executable
$pythonExe = (Get-Command py -ErrorAction SilentlyContinue)
if (-not $pythonExe) {
    $pythonExe = (Get-Command python -ErrorAction SilentlyContinue)
}
if (-not $pythonExe) {
    Write-Error "Python executable not found in PATH."
    exit 1
}

# 2. Check CUDA availability
Write-Host "Verifying PyTorch CUDA..." -ForegroundColor Yellow
$cudaCheck = & $pythonExe -c "import torch; print('CUDA:' + str(torch.cuda.is_available()) + ', Device:' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'))"
Write-Host "  $cudaCheck"

# 3. Assemble command arguments
$cmdArgs = @(
    "scripts/train_lora_qwen.py",
    "--model_id", $ModelId,
    "--train_dataset", $TrainDataset,
    "--output_dir", $OutputDir,
    "--people_prompt_path", $PeoplePromptPath,
    "--decision_prompt_path", $DecisionPromptPath,
    "--description_prompt_path", $DescriptionPromptPath,
    "--method_filter", $MethodFilter,
    "--quantization", $Quantization,
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--grad_accum", $GradAccum,
    "--learning_rate", $LearningRate,
    "--lora_r", $LoraR,
    "--lora_alpha", $LoraAlpha
)

if ($MergeAndSave) {
    $cmdArgs += "--merge_and_save"
}

if ($DryRun) {
    $cmdArgs += "--dry_run"
}

# 4. Launch training
Write-Host "`nLaunching training process..." -ForegroundColor Green
$startTime = Get-Date

& $pythonExe @cmdArgs

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`n=== Process Completed ===" -ForegroundColor Cyan
Write-Host "Elapsed Time: $($duration.ToString('hh\:mm\:ss'))"
if (-not $DryRun) {
    Write-Host "LoRA adapter saved to: $OutputDir" -ForegroundColor Green
    if ($MergeAndSave) {
        Write-Host "Merged standalone model saved to: ${OutputDir}-merged" -ForegroundColor Green
    }
}
