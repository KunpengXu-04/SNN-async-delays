import fs from "fs";
import path from "path";

const ARTIFACT_TOOL =
  "file:///C:/Users/xukun/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const { Presentation, PresentationFile } = await import(ARTIFACT_TOOL);

const ROOT = path.resolve("D:/xukun/Documents/IC/SNN/SNN_project");
const BASE_PATH = path.join(ROOT, "presentation.pptx");
const STEP4_PATH = path.join(ROOT, "presentation_step4_update.pptx");
const FLOW_PATH = path.join(ROOT, "presentation_topology_flowcharts.pptx");
const BACKUP_PATH = path.join(ROOT, "presentation_before_step4_merge_backup.pptx");

function loadDeckProto(filePath) {
  const bytes = fs.readFileSync(filePath);
  return PresentationFile.importPptx(bytes).then((deck) => deck.toProto());
}

function byteArrayEquals(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function uniqueId(existing, base) {
  let candidate = base;
  let index = 1;
  while (existing.has(candidate)) {
    candidate = `${base}_${index}`;
    index += 1;
  }
  existing.add(candidate);
  return candidate;
}

function nextImageId(existingImageIds, originalId, tag) {
  const parsed = path.posix.parse(originalId);
  const stem = parsed.name || "image";
  const ext = parsed.ext || ".png";
  let candidate = `${parsed.dir}/${tag}_${stem}${ext}`;
  let index = 1;
  while (existingImageIds.has(candidate)) {
    candidate = `${parsed.dir}/${tag}_${stem}_${index}${ext}`;
    index += 1;
  }
  existingImageIds.add(candidate);
  return candidate;
}

function walkMutate(node, fn) {
  if (!node || typeof node !== "object") return;
  if (Array.isArray(node)) {
    node.forEach((child) => walkMutate(child, fn));
    return;
  }
  fn(node);
  for (const value of Object.values(node)) {
    walkMutate(value, fn);
  }
}

function appendDeckPart(targetProto, sourceProto, slides, tag) {
  const slideIds = new Set(targetProto.slides.map((slide) => slide.id));
  const creationIds = new Set(
    targetProto.slides.map((slide) => slide.creationId).filter(Boolean),
  );
  const existingImageIds = new Set(targetProto.images.map((img) => img.id));
  const targetImagesById = new Map(targetProto.images.map((img) => [img.id, img]));
  const sourceImageIdMap = new Map();

  for (const image of sourceProto.images ?? []) {
    const existing = targetImagesById.get(image.id);
    if (existing && byteArrayEquals(existing.data, image.data)) {
      sourceImageIdMap.set(image.id, image.id);
      continue;
    }

    let finalId = image.id;
    if (existingImageIds.has(finalId)) {
      finalId = nextImageId(existingImageIds, image.id, tag);
    } else {
      existingImageIds.add(finalId);
    }

    sourceImageIdMap.set(image.id, finalId);
    targetProto.images.push({ ...image, id: finalId });
    targetImagesById.set(finalId, { ...image, id: finalId });
  }

  for (const originalSlide of slides) {
    const slide = structuredClone(originalSlide);
    slide.id = uniqueId(slideIds, `${tag}_${slide.id}`);
    slide.index = targetProto.slides.length;

    if (slide.creationId && creationIds.has(slide.creationId)) {
      let suffix = 1;
      let nextCreationId = `${slide.creationId}_${suffix}`;
      while (creationIds.has(nextCreationId)) {
        suffix += 1;
        nextCreationId = `${slide.creationId}_${suffix}`;
      }
      slide.creationId = nextCreationId;
    }
    if (slide.creationId) creationIds.add(slide.creationId);

    walkMutate(slide, (node) => {
      if (node.imageReference?.id && sourceImageIdMap.has(node.imageReference.id)) {
        node.imageReference.id = sourceImageIdMap.get(node.imageReference.id);
      }
    });

    targetProto.slides.push(slide);
  }
}

function extractTitles(proto) {
  return proto.slides.map((slide, idx) => {
    const text = [];
    for (const element of slide.elements ?? []) {
      for (const paragraph of element.paragraphs ?? []) {
        for (const run of paragraph.runs ?? []) {
          if (run.text?.trim()) text.push(run.text.trim());
        }
      }
      if (text.length > 0) break;
    }
    return `${idx + 1}. ${text[0] ?? "(untitled)"}`;
  });
}

const [baseProto, step4Proto, flowProto] = await Promise.all([
  loadDeckProto(BASE_PATH),
  loadDeckProto(STEP4_PATH),
  loadDeckProto(FLOW_PATH),
]);

if (!fs.existsSync(BACKUP_PATH)) {
  fs.copyFileSync(BASE_PATH, BACKUP_PATH);
}

const merged = structuredClone(baseProto);

appendDeckPart(merged, step4Proto, step4Proto.slides.slice(0, 3), "step4a");
appendDeckPart(merged, flowProto, flowProto.slides, "flow");
appendDeckPart(merged, step4Proto, step4Proto.slides.slice(3), "step4b");

const mergedDeck = Presentation.load(merged);
const blob = await PresentationFile.exportPptx(mergedDeck);
fs.writeFileSync(BASE_PATH, Buffer.from(blob.data));

console.log(
  JSON.stringify(
    {
      output: BASE_PATH,
      backup: BACKUP_PATH,
      slides: merged.slides.length,
      titles: extractTitles(merged),
    },
    null,
    2,
  ),
);
