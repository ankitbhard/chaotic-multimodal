#!/usr/bin/env bash
# Inspect S3 and stage existing UNI embeddings and gene CSVs for the isolated pilot.
set -euo pipefail
REGION="${AWS_REGION:-us-east-2}"
BUCKET="${S3_BUCKET:-chaotic-tangle-brca}"
echo "Bucket: s3://${BUCKET}/ (region ${REGION})"
echo "Top-level S3 objects/prefixes:"
aws s3 ls "s3://${BUCKET}/" --region "${REGION}"
if [[ "${1:-}" == "--list" ]]; then
  echo "Sample objects (first 80):"
  aws s3api list-objects-v2 --bucket "${BUCKET}" --region "${REGION}" --max-items 80 --query 'Contents[].Key' --output text
  exit 0
fi
if [[ -z "${UNI_S3_PREFIX:-}" || -z "${GENE_S3_PREFIX:-}" ]]; then
  echo "Set UNI_S3_PREFIX and GENE_S3_PREFIX (S3 prefixes relative to bucket root), or use --list first." >&2
  exit 2
fi
UNI_LOCAL="${UNI_LOCAL:-/home/ubuntu/tangle_data/brca/brca}"
GENE_LOCAL="${GENE_LOCAL:-/home/ubuntu/tcga_brca_data}"
mkdir -p "${UNI_LOCAL}" "${GENE_LOCAL}"
aws s3 sync "s3://${BUCKET}/${UNI_S3_PREFIX#/}" "${UNI_LOCAL}" --region "${REGION}" --exclude "*" --include "*.pt" --include "*.h5"
aws s3 sync "s3://${BUCKET}/${GENE_S3_PREFIX#/}" "${GENE_LOCAL}" --region "${REGION}" --exclude "*" --include "*.csv"
echo "Staging finished. Verify the YAML paths with run_pilot.py --dry-run."
