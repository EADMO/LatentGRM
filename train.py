"""Run the Semantic Chunking training recipe, with automatic stage recovery."""
from pathlib import Path
import argparse, json, math, os, shlex, shutil, subprocess, sys

ROOT=Path(__file__).resolve().parent
PHASES=['cache','encoder','decoder','joint','targets','merge','stage2']

def read(path): return json.loads(Path(path).read_text())
def nonempty(path): return path.is_file() and path.stat().st_size>0
def count_records(path):
    with path.open(encoding='utf-8') as stream:
        records=sum(bool(line.strip()) for line in stream)
    if not records: raise ValueError(f'Training data is empty: {path}')
    return records
def hf_complete(path):
    if not nonempty(path/'config.json'): return False
    indices=list(path.glob('*.index.json'))
    if indices:
        return all(nonempty(path/f) for index in indices for f in set(read(index)['weight_map'].values()))
    return any(nonempty(p) for p in [*path.glob('*.safetensors'),*path.glob('pytorch_model*.bin')])

def checkpoint_complete(path,world):
    try:
        state=read(path/'trainer_state.json');step=int(state['global_step'])
        if path.name!=f'checkpoint-{step}': return False
        ds=path/f'global_step{step}'
        if not nonempty(ds/'mp_rank_00_model_states.pt'): return False
        for rank in range(world):
            if not nonempty(ds/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'): return False
            rng=path/('rng_state.pth' if world==1 else f'rng_state_{rank}.pth')
            if not nonempty(rng): return False
        return nonempty(path/'latest')
    except (OSError,ValueError,KeyError): return False

def latest_checkpoint(directory,world):
    paths=[p for p in directory.glob('checkpoint-*') if p.name.removeprefix('checkpoint-').isdigit()]
    return next((p for p in sorted(paths,key=lambda p:int(p.name.split('-')[-1]),reverse=True) if checkpoint_complete(p,world)),None)

def validate_config(c,records):
    steps=math.ceil(math.ceil(records/c['world_size'])/c['stage2_batch_size'])
    steps=math.ceil(steps/c['stage2_accumulation'])
    if c['stage2_epochs']<1: raise ValueError('Training epochs must be positive.')
    if not 0<c['stage2_warmup_steps']<c['stage2_lr_decay_steps']:
        raise ValueError('Expected 0 < warmup steps < learning-rate decay steps.')
    return steps

def targets_complete(path,records):
    try:
        manifest=read(path/'manifest.json')
        if manifest['num_records']!=records or not (path/'.completed').exists(): return False
        size=manifest['chunk_size']
        return all(nonempty(path/f'batch_{i}_{min(i+size,records)}.pt') for i in range(0,records,size))
    except (OSError,ValueError,KeyError): return False

def stage_export_complete(path,phase):
    adapter=path/'lora_adapter'
    if phase=='joint':
        return all(nonempty(adapter/x/'adapter_config.json') and nonempty(adapter/x/'adapter_model.safetensors') for x in ['encoder_weight','decoder_weight'])
    return hf_complete(path/'hf') and nonempty(adapter/'adapter_config.json') and nonempty(adapter/'adapter_model.safetensors')

def promote_checkpoint(checkpoint,path,phase):
    if not stage_export_complete(checkpoint,phase):
        raise RuntimeError('Final checkpoint exports are incomplete; resume from the preceding complete checkpoint.')
    for name in ['lora_adapter']+([] if phase=='joint' else ['hf']):
        shutil.copytree(checkpoint/name,path/name,dirs_exist_ok=True)
    shutil.copy2(checkpoint/'trainer_state.json',path/'trainer_state.json')
    if not stage_export_complete(path,phase): raise RuntimeError('Final export validation failed.')
    (path/'.completed').touch()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/qwen3_8b.json')
    parser.add_argument('--stage',choices=['all',*PHASES],default='all')
    parser.add_argument('--output')
    parser.add_argument('--model')
    parser.add_argument('--resume',type=Path,help='Complete checkpoint for the selected training stage.')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();os.chdir(ROOT)
    c=read(args.config)
    if args.output: c['output']=args.output
    if args.model: c['model']=args.model
    if args.resume and args.stage not in ['encoder','decoder','joint','stage2']:
        parser.error('--resume requires a specific training stage; all resumes automatically.')
    out=Path(c['output']).resolve();model=Path(c['model']).resolve();data=Path(c['data']).resolve()
    records=count_records(data)
    steps=validate_config(c,records)
    c['records']=records
    env=dict(os.environ);env['TOKENIZERS_PARALLELISM']='false';env.setdefault('OMP_NUM_THREADS','1')
    env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    env.setdefault('HF_HUB_DISABLE_TELEMETRY','1');env.setdefault('WANDB_DISABLED','true')
    stage_dirs={s:out/s for s in ['encoder','decoder','joint','stage2']}
    cache=out/'cache';labels=out/'targets';merged=out/'joint_decoder'
    if not args.dry_run:
        out.mkdir(parents=True,exist_ok=True)
        (out/'run.json').write_text(json.dumps(c,indent=2)+'\n')
        snapshot=out/'code'
        if not snapshot.exists():
            shutil.copytree(ROOT/'latentgrm',snapshot/'latentgrm',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(ROOT/'configs',snapshot/'configs')
            shutil.copy2(__file__,snapshot/'train.py')
            (snapshot/'command.txt').write_text(shlex.join(sys.argv)+'\n')
            freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
            (snapshot/'environment.txt').write_text(freeze)
    def run(command):
        print(shlex.join(map(str,command)),flush=True)
        if not args.dry_run: subprocess.run(list(map(str,command)),env=env,check=True)
    def module(name,*argv): return [sys.executable,'-m',name,*argv]
    def distributed(name):
        return [sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={c["world_size"]}','--module',name]
    def require_stage(phase):
        if args.dry_run: return
        path=stage_dirs[phase]
        if not (path/'.completed').exists() or not stage_export_complete(path,phase):
            raise RuntimeError(f'Complete the {phase} stage first.')
    def training(phase):
        path=stage_dirs[phase]
        for prerequisite in {'encoder':[], 'decoder':['encoder'], 'joint':['encoder','decoder'], 'stage2':['joint']}[phase]:
            require_stage(prerequisite)
        if phase=='stage2' and not args.dry_run:
            if not targets_complete(labels,records) or not hf_complete(merged/'hf'):
                raise RuntimeError('Complete targets and merge before Stage 2.')
        if phase=='stage2':
            epochs=c['stage2_epochs'];total_steps=steps*epochs;lr=c['stage2_learning_rate'];batch=c['stage2_batch_size'];accum=c['stage2_accumulation']
            ds_path=out/'stage2_deepspeed.json'
            if not args.dry_run:
                ds=read(c['deepspeed'])
                # The Trainer supplies WarmupDecayLR with an explicit decay scale.
                ds.pop('scheduler',None)
                ds_path.write_text(json.dumps(ds,indent=2)+'\n')
            command=distributed('latentgrm.semantic_chunking.run_stage2')+['--latent_model_path',merged/'hf','--train_latent_soft_label_path',labels,'--ce_w','1','--kl_w','1','--training','True','--add_gumbel_noise','True','--gumbel_temperature','1','--noise_scale','1','--use_flash_attention_2','True']
            command+=['--lr_decay_steps',c['stage2_lr_decay_steps'],'--warmup_steps',c['stage2_warmup_steps']]
        else:
            ds_path=c['deepspeed']
            minibatches=math.ceil(math.ceil(records/c['world_size'])/c['stage1_batch_size'])
            total_steps=math.ceil(minibatches/c['stage1_accumulation'])*c['stage1_epochs']
            epochs=c['stage1_epochs'];lr=c['stage1_learning_rate'];batch=c['stage1_batch_size'];accum=c['stage1_accumulation']
            encoder=model if phase=='encoder' else stage_dirs['encoder']/'hf'
            decoder=stage_dirs['decoder']/'hf' if phase=='joint' else model
            command=distributed('latentgrm.semantic_chunking.run_stage1')+['--stage','union' if phase=='joint' else phase,'--use','semantic','--encoder_name_or_path',encoder,'--decoder_name_or_path',decoder,'--stage1_cache_path',cache,'--compression_rate',c['compression_rate'],'--use_flash_attention_2','False','--dataloader_num_workers','8','--dataloader_prefetch_factor','16','--dataloader_pin_memory','True']
        if (path/'.completed').exists():
            state=read(path/'trainer_state.json')
            valid=stage_export_complete(path,phase)
            if state['global_step']!=total_steps or not valid: raise RuntimeError(f'Incomplete completed stage: {phase}')
            print(f'{phase}: complete');return
        resume=args.resume.resolve() if args.resume else latest_checkpoint(path,c['world_size'])
        if resume and not checkpoint_complete(resume,c['world_size']): raise ValueError(f'Incomplete resume checkpoint: {resume}')
        if resume and resume.parent!=path: raise ValueError('--resume must belong to this stage and output directory.')
        if resume and read(resume/'trainer_state.json')['global_step']>=total_steps:
            if read(resume/'trainer_state.json')['global_step']!=total_steps:
                raise RuntimeError('Checkpoint is beyond the configured training target.')
            if not args.dry_run: promote_checkpoint(resume,path,phase)
            print(f'{phase}: recover final export from {resume.name}');return
        command+=['--train_data_path',data,'--bfloat16','True','--topk_interpolation',c['topk'],'--lora_tune','True','--lora_rank',c['lora_rank'],'--lora_dropout',c['lora_dropout'],'--deepspeed',ds_path,'--no_remove_unused_columns','--learning_rate',lr,'--warmup_ratio','0' if phase=='stage2' else '0.05','--weight_decay','0.01','--num_train_epochs',epochs,'--bf16','--per_device_train_batch_size',batch,'--gradient_accumulation_steps',accum,'--dataloader_drop_last','False','--logging_steps','10','--save_total_limit','2','--save_strategy','epoch','--gradient_checkpointing','True','--report_to','none','--seed',c['seed'],'--output_dir',path]
        if resume: command+=['--resume_from_checkpoint',resume]
        run(command)
        if not args.dry_run:
            if read(path/'trainer_state.json')['global_step']!=total_steps: raise RuntimeError('Training stopped before its target.')
            if not stage_export_complete(path,phase): raise RuntimeError(f'Incomplete final export: {phase}')
            (path/'.completed').touch()
    print(f'Stage 2: {c["stage2_epochs"]} epochs, {steps} optimizer steps per epoch',flush=True)
    for phase in PHASES if args.stage=='all' else [args.stage]:
        if phase=='cache':
            run(module('latentgrm.semantic_chunking.build_stage1_cache','--data',data,'--tokenizer',model,'--output',cache,'--use','semantic','--compression-rate',c['compression_rate'],'--workers',c['cache_workers']))
        elif phase in stage_dirs: training(phase)
        elif phase=='targets':
            require_stage('joint')
            if targets_complete(labels,records): print('targets: complete');continue
            run(module('latentgrm.semantic_chunking.generate_soft_labels','--encoder_model_path',stage_dirs['encoder']/'hf','--decoder_model_path',stage_dirs['decoder']/'hf','--lora_path',stage_dirs['joint']/'lora_adapter','--save_path',labels,'--data_path',data,'--use','semantic','--stage1_cache_path',cache,'--mp_size',c['world_size'],'--batch_size','16','--chunk_size','1000','--dtype','bfloat16','--compression_rate',c['compression_rate'],'--topk_interpolation',c['topk'],'--resume'))
            if not args.dry_run and not targets_complete(labels,records): raise RuntimeError('Soft target export is incomplete.')
        elif phase=='merge':
            require_stage('joint')
            if hf_complete(merged/'hf'): print('joint decoder: complete');continue
            run(module('latentgrm.training.merge','--base_model_path',stage_dirs['decoder']/'hf','--lora_path',stage_dirs['joint']/'lora_adapter/decoder_weight','--output_path',merged,'--output_subdir','hf','--dtype','bfloat16','--attn_implementation','sdpa','--device_map','auto'))
            if not args.dry_run and not hf_complete(merged/'hf'): raise RuntimeError('Joint decoder merge is incomplete.')

if __name__=='__main__': main()
