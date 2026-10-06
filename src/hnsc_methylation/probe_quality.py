#!/usr/bin/env python3
"""Apply an annotation-only quality rule, independently of study associations.

The source masks are sparse: a row's absence means no listed flag ONLY after
coverage of its probe ID is confirmed in that platform's complete manifest.
For this mixed-platform matrix we conservatively take the union of HM27 and
HM450 flags, requiring confirmed manifest coverage on both platforms.
"""
from pathlib import Path
import hashlib
import pandas as pd

REPOSITORY = 'zhou-lab/InfiniumAnnotationData'
COMMIT = '6d176d6e7e56465403d4478a09fe11b983eeb721'
FILES = {
    'HM27': {'manifest': 'HM27.hg38.manifest.tsv.gz', 'mask': 'HM27.hg38.mask.tsv.gz'},
    'HM450': {'manifest': 'HM450.hg19.manifest.tsv.gz', 'mask': 'HM450.hg38.mask.tsv.gz'},
}
EXPECTED_GIT_BLOBS = {
    'HM27.hg38.manifest.tsv.gz': '1ede3bc10ca105053d28a39cdf3199c3633aa43f',
    'HM27.hg38.mask.tsv.gz': 'd3546b3c8831df7a2fec8f52435c3c02119ce39f',
    'HM450.hg19.manifest.tsv.gz': '71ac7ff6d861ed5d7a452e4917e6129998089b0e',
    'HM450.hg38.mask.tsv.gz': '024d42f3a3d88698f1b5a0c981775d535e3eb4a8',
}
POPULATION_TAGS = {
    f'{prefix}_{population}_1pt'
    for prefix in ['M_SNP', 'M_1baseSwitchSNP', 'M_2extBase_SNP']
    for population in ['AFR', 'EUR']
}


def file_metadata(path, platform):
    contents = path.read_bytes()
    blob_sha1 = hashlib.sha1(b'blob '+str(len(contents)).encode()+b'\0'+contents).hexdigest()
    assert blob_sha1 == EXPECTED_GIT_BLOBS[path.name], 'Local source differs from verified pinned repository file'
    return {
        'file': path.name,
        'url': f'https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/Anno/{platform}/{path.name}',
        'sha256': hashlib.sha256(contents).hexdigest(),
        'git_blob_sha1': blob_sha1,
        'pinned_repository_blob_verified': True,
    }


def annotate_probe_quality(meta: pd.DataFrame, annotation_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Apply the pinned manifest/mask rule directly to source annotations."""
    original = meta[['Entity_Stable_Id', 'HugoSymbol', 'Location']].copy()
    annotation_dir = Path(annotation_dir)
    out = original.groupby('Entity_Stable_Id', sort=False).agg({
        'HugoSymbol': lambda s: '/'.join(dict.fromkeys(s.dropna().astype(str))),
        'Location': lambda s: '/'.join(dict.fromkeys(s.dropna().astype(str))),
    }).reset_index()
    assert out.Entity_Stable_Id.is_unique
    source_metadata = []
    for platform, names in FILES.items():
        manifest_path = annotation_dir/names['manifest']
        mask_path = annotation_dir/names['mask']
        ids = pd.read_csv(manifest_path, sep='\t', usecols=['Probe_ID']).Probe_ID
        masks = pd.read_csv(mask_path, sep='\t').set_index('Probe_ID')
        assert ids.is_unique and masks.index.is_unique
        assert set(masks.index).issubset(set(ids)), 'Mask contains IDs absent from manifest'
        coverage = out.Entity_Stable_Id.isin(ids)
        out[f'{platform}_manifest_covered'] = coverage
        out[f'{platform}_mask_row_present'] = out.Entity_Stable_Id.isin(masks.index)
        for column in ['mask', 'maskUniq']:
            out[f'{platform}_{column}'] = out.Entity_Stable_Id.map(masks[column]).fillna('')
        # Missing sparse-mask entries are clear only within the known universe.
        out[f'{platform}_general_mask'] = out.Entity_Stable_Id.isin(masks.index[masks.M_general])
        out[f'{platform}_population_SNP_mask'] = out[f'{platform}_maskUniq'].map(
            lambda text: bool(set(text.split(',')) & POPULATION_TAGS)
        )
        out[f'{platform}_annotation_status'] = [
            'covered_flagged' if row_present else 'covered_no_listed_flags' if covered else 'not_covered'
            for covered, row_present in zip(coverage, out[f'{platform}_mask_row_present'])
        ]
        source_metadata += [file_metadata(manifest_path, platform), file_metadata(mask_path, platform)]
    out['covered_both_platforms'] = out.HM27_manifest_covered & out.HM450_manifest_covered
    out['general_mask_union'] = out.HM27_general_mask | out.HM450_general_mask
    out['population_SNP_mask_union'] = out.HM27_population_SNP_mask | out.HM450_population_SNP_mask
    out['technical_mask_union'] = out.general_mask_union | out.population_SNP_mask_union
    out['eligible_primary'] = out.covered_both_platforms & ~out.technical_mask_union
    out['exclusion_reason'] = [
        ';'.join(reason for reason, present in [
            ('manifest_coverage_incomplete', not covered),
            ('general_quality_mask', general),
            ('AFR_or_EUR_SNP_at_least_1_percent', population),
        ] if present)
        for covered, general, population in zip(out.covered_both_platforms, out.general_mask_union, out.population_SNP_mask_union)
    ]
    summary = {
        'annotation_repository': REPOSITORY,
        'annotation_repository_commit': COMMIT,
        'repository_commit_date': '2026-07-19T20:47:30Z',
        'retrieval_date': '2026-09-04',
        'study_probe_universe_source': 'data/inputs/methylation.csv.gz',
        'mask_coordinate_assembly': 'hg38 for both platforms',
        'manifest_usage': 'Probe-ID coverage only: HM27 hg38 and HM450 hg19. No genomic coordinates are joined across assemblies.',
        'rule': 'Require coverage in both full manifests; exclude union of M_general=True or any of the six AFR/EUR SNP tags below in either platform. Rule uses no differential methylation P value, effect size, survival or expression evidence.',
        'population_SNP_tags': sorted(POPULATION_TAGS),
        'population_SNP_definition': 'Published 1pt tags flag population minor allele frequency at least 1%: SNP within 5 bp of the 3-prime extension, an Infinium-I colour-channel-switch SNP, or an Infinium-II extension-base SNP. The general mask is used exactly as supplied.',
        'sparse_mask_interpretation': 'Absence of a row is interpreted as no listed mask only after manifest membership is confirmed; unknown coverage is excluded, never assumed clear.',
        'reference_pmid': '27924034',
        'reference_doi': '10.1093/nar/gkw967',
        'reference_url': 'https://pubmed.ncbi.nlm.nih.gov/27924034/',
        'annotation_documentation_url': 'https://zwdzwd.github.io/InfiniumAnnotation/',
        'sources': source_metadata,
        'counts': {
            'study_probes': len(out),
            'covered_HM27': int(out.HM27_manifest_covered.sum()),
            'covered_HM450': int(out.HM450_manifest_covered.sum()),
            'general_mask_union': int(out.general_mask_union.sum()),
            'population_SNP_mask_union': int(out.population_SNP_mask_union.sum()),
            'overlap_general_and_population': int((out.general_mask_union & out.population_SNP_mask_union).sum()),
            'technical_exclusions': int(out.technical_mask_union.sum()),
            'unknown_coverage': int((~out.covered_both_platforms).sum()),
            'eligible_primary': int(out.eligible_primary.sum()),
        },
    }
    return out, summary
