#![allow(clippy::type_complexity)]
// UltralyticsYOLO Model Processing - Advanced AI inference engine
// These warnings are suppressed to preserve planned architecture for future development

use anyhow::Result;
use image::{DynamicImage, ImageBuffer};
use ndarray::{s, Array, Axis, IxDyn};
use opencv::prelude::*;
use rand::{thread_rng, Rng};

use crate::{
    non_max_suppression, Batch, Bbox, Embedding, OrtBackend,
    OrtConfig, OrtEP, Point2, YOLOResult, YOLOTask,
};


pub struct UltralyticsYOLO {
    // UltralyticsYOLO model for all yolo-tasks
    pub engine: OrtBackend,
    pub nc: u32,
    pub nk: u32,
    pub nm: u32,
    pub height: u32,
    pub width: u32,
    pub batch: u32,
    pub task: YOLOTask,
    pub conf: f32,
    pub kconf: f32,
    pub iou: f32,
    pub names: Vec<String>,
    pub color_palette: Vec<(u8, u8, u8)>,
    pub profile: bool,
}

impl UltralyticsYOLO {
    pub fn new(config: crate::config::Config) -> Result<Self> {
        // execution provider
        let ep = if config.trt {
            OrtEP::Trt(config.device_id)
        } else if config.cuda {
            OrtEP::Cuda(config.device_id)
        } else {
            OrtEP::Cpu
        };

        // batch
        let batch = Batch {
            opt: config.batch,
            min: config.batch_min,
            max: config.batch_max,
        };

        // build ort engine
        let ort_args = OrtConfig {
            ep,
            batch,
            f: config.model,
            task: None, // Auto-detect from model
            trt_fp16: config.fp16,
            image_size: (Some(config.height), Some(config.width)),
        };
        let engine = OrtBackend::build(ort_args)?;

        //  get batch, height, width, tasks, nc, nk, nm
        let (batch, height, width, task) = (
            engine.batch(),
            engine.height(),
            engine.width(),
            engine.task(),
        );
        let nc = engine.nc().or(config.nc).unwrap_or_else(|| {
            panic!("Failed to get num_classes, make it explicit with `--nc`");
        });
        let (nk, nm) = match task {
            YOLOTask::Pose => {
                let nk = engine.nk().or(config.nk).unwrap_or_else(|| {
                    panic!("Failed to get num_keypoints, make it explicit with `--nk`");
                });
                (nk, 0)
            }
            YOLOTask::Segment => {
                let nm = engine.nm().or(config.nm).unwrap_or_else(|| {
                    panic!("Failed to get num_masks, make it explicit with `--nm`");
                });
                (0, nm)
            }
            _ => (0, 0),
        };

        // class names
        let names = engine.names().unwrap_or(vec!["Unknown".to_string()]);

        // color palette
        let mut rng = thread_rng();
        let color_palette: Vec<_> = names
            .iter()
            .map(|_| {
                (
                    rng.gen_range(0..=255),
                    rng.gen_range(0..=255),
                    rng.gen_range(0..=255),
                )
            })
            .collect();

        Ok(Self {
            engine,
            names,
            conf: config.conf,
            kconf: config.kconf,
            iou: config.iou,
            color_palette,
            profile: config.profile,
            nc,
            nk,
            nm,
            height,
            width,
            batch,
            task,
        })
    }

    pub fn scale_wh(&self, w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        let r = (w1 / w0).min(h1 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }



    pub fn run_mat(&mut self, xs: &Vec<opencv::core::Mat>) -> Result<Vec<YOLOResult>> {
        // pre-process
        let t_pre = std::time::Instant::now();

        // Convert cv::Mat to f32 array directly
        let mut xs_array =
            Array::ones((xs.len(), 3, self.height as usize, self.width as usize)).into_dyn();
        xs_array.fill(114.0 / 255.0); // Fill with gray padding color (114)

        let idx = 0; // Set idx to 0
        let mat = &xs[idx]; // Get the current Mat from the vector
        
        // Store original dimensions for scaling
        let original_width = mat.cols() as u32;
        let original_height = mat.rows() as u32;
        
        // Calculate scale to maintain aspect ratio (match preprocess method)
        let (_ratio, new_width, new_height) = self.scale_wh(
            original_width as f32,
            original_height as f32,
            self.width as f32,
            self.height as f32
        );
                            
        // Resize the image while maintaining aspect ratio
        let mut resized = opencv::core::Mat::default();
        opencv::imgproc::resize(
            mat,
            &mut resized,
            opencv::core::Size::new(new_width as i32, new_height as i32),
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;

        // Convert BGR to RGB
        let bgr = resized.clone();
        let mut channels: opencv::core::Vector<opencv::core::Mat> = opencv::core::Vector::new();
        opencv::core::split(&bgr, &mut channels)?;
        
        // Swap B and R channels
        if channels.len() >= 3 {
            let temp = channels.get(0)?;
            channels.set(0, channels.get(2)?)?;
            channels.set(2, temp)?;
        }
        
        let mut rgb = opencv::core::Mat::default();
        opencv::core::merge(&channels, &mut rgb)?;

        // Calculate padding offsets to center the resized image
        let pad_left = ((self.width as f32 - new_width) / 2.0).floor() as i32;
        let pad_top = ((self.height as f32 - new_height) / 2.0).floor() as i32;
        
        // Copy resized image data into padded array
        for y in 0..new_height as i32 {
            for x in 0..new_width as i32 {
                if y < rgb.rows() && x < rgb.cols() {
                    let pixel = rgb.at_2d::<opencv::core::Vec3b>(y, x)?;
                    let target_y = (y + pad_top) as usize;
                    let target_x = (x + pad_left) as usize;
                    
                    if target_y < self.height as usize && target_x < self.width as usize {
                        xs_array[[idx, 0, target_y, target_x]] = pixel[0] as f32 / 255.0;
                        xs_array[[idx, 1, target_y, target_x]] = pixel[1] as f32 / 255.0;
                        xs_array[[idx, 2, target_y, target_x]] = pixel[2] as f32 / 255.0;
                    }
                }
            }
        }

        if self.profile {
            println!("[Model Preprocess]: {:?}", t_pre.elapsed());
        }

        // run
        let t_run = std::time::Instant::now();
        let ys = self.engine.run(xs_array, self.profile)?;
        if self.profile {
            println!("[Model Inference]: {:?}", t_run.elapsed());
        }

        // post-process
        let t_post = std::time::Instant::now();
        // Create dummy DynamicImage objects with original dimensions for postprocessing
        let dummy_images: Vec<DynamicImage> = xs
            .iter()
            .map(|mat| DynamicImage::new_rgb8(mat.cols() as u32, mat.rows() as u32))
            .collect();
        let ys = self.postprocess(ys, &dummy_images)?;
        if self.profile {
            println!("[Model Postprocess]: {:?}", t_post.elapsed());
        }

        Ok(ys)
    }


    pub fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>,
        xs0: &[DynamicImage],
    ) -> Result<Vec<YOLOResult>> {
        if let YOLOTask::Classify = self.task {
            let mut ys = Vec::new();
            let preds = &xs[0];
            for batch in preds.axis_iter(Axis(0)) {
                ys.push(YOLOResult::new(
                    Some(Embedding::new(batch.into_owned())),
                    None,
                    None,
                    None,
                ));
            }
            Ok(ys)
        } else {
            const CXYWH_OFFSET: usize = 4; // cxcywh
            const KPT_STEP: usize = 3; // xyconf
            let preds = &xs[0];
            let protos = {
                if xs.len() > 1 {
                    Some(&xs[1])
                } else {
                    None
                }
            };
            let mut ys = Vec::new();
            for (idx, anchor) in preds.axis_iter(Axis(0)).enumerate() {
                // [bs, 4 + nc + nm, anchors]
                // input image
                let width_original = xs0[idx].width() as f32;
                let height_original = xs0[idx].height() as f32;
                
                // Calculate scaling ratio with improved precision
                let width_ratio = self.width as f32 / width_original;
                let height_ratio = self.height as f32 / height_original;
                let ratio = width_ratio.min(height_ratio);
                
                // Calculate actual dimensions used in the model (accounting for padding)
                let model_width = (width_original * ratio) as f32;
                let model_height = (height_original * ratio) as f32;
                
                // Calculate padding offsets (if any)
                let padx = (self.width as f32 - model_width) / 2.0;
                let pady = (self.height as f32 - model_height) / 2.0;

                // save each result
                let mut data: Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> = Vec::new();
                for pred in anchor.axis_iter(Axis(1)) {
                    // split preds for different tasks
                    let bbox = pred.slice(s![0..CXYWH_OFFSET]);
                    let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc as usize]);
                    let kpts = {
                        if let YOLOTask::Pose = self.task {
                            Some(pred.slice(s![pred.len() - KPT_STEP * self.nk as usize..]))
                        } else {
                            None
                        }
                    };
                    let coefs = {
                        if let YOLOTask::Segment = self.task {
                            Some(pred.slice(s![pred.len() - self.nm as usize..]).to_vec())
                        } else {
                            None
                        }
                    };

                    // confidence and id
                    let (id, &confidence) = clss
                        .into_iter()
                        .enumerate()
                        .reduce(|max, x| if x.1 > max.1 { x } else { max })
                        .unwrap(); // definitely will not panic!

                    // confidence filter
                    if confidence < self.conf {
                        continue;
                    }

                    // Adjust for padding if any
                    let cx = (bbox[0] - padx) / ratio;
                    let cy = (bbox[1] - pady) / ratio;
                    let w = bbox[2] / ratio;
                    let h = bbox[3] / ratio;
                    let x = cx - w / 2.;
                    let y = cy - h / 2.;
                    
                    // Ensure coordinates are within image bounds
                    let x_min = x.max(0.0).min(width_original);
                    let y_min = y.max(0.0).min(height_original);
                    let y_bbox = Bbox::new(
                        x_min,
                        y_min,
                        w.min(width_original - x_min),
                        h.min(height_original - y_min),
                        id,
                        confidence,
                    );

                    // kpts
                    let y_kpts = {
                        if let Some(kpts) = kpts {
                            let mut kpts_ = Vec::new();
                            // rescale keypoints with the same ratio
                            for i in 0..self.nk as usize {
                                let kx = (kpts[KPT_STEP * i] - padx) / ratio;
                                let ky = (kpts[KPT_STEP * i + 1] - pady) / ratio;
                                let kconf = kpts[KPT_STEP * i + 2];
                                if kconf < self.kconf {
                                    kpts_.push(Point2::default());
                                } else {
                                    kpts_.push(Point2::new_with_conf(
                                        kx.max(0.0f32).min(width_original),
                                        ky.max(0.0f32).min(height_original),
                                        kconf,
                                    ));
                                }
                            }
                            Some(kpts_)
                        } else {
                            None
                        }
                    };

                    // data merged
                    data.push((y_bbox, y_kpts, coefs));
                }

                // nms
                non_max_suppression(&mut data, self.iou);

                // decode
                let mut y_bboxes: Vec<Bbox> = Vec::new();
                let mut y_kpts: Vec<Vec<Point2>> = Vec::new();
                let mut y_masks: Vec<Vec<u8>> = Vec::new();
                for elem in data.into_iter() {
                    if let Some(kpts) = elem.1 {
                        y_kpts.push(kpts)
                    }

                    // decode masks
                    if let Some(coefs) = elem.2 {
                        let proto = protos.unwrap().slice(s![idx, .., .., ..]);
                        let (nm, nh, nw) = proto.dim();

                        // coefs * proto -> mask
                        let coefs = Array::from_shape_vec((1, nm), coefs)?; // (n, nm)
                        let proto = proto.to_owned().into_shape((nm, nh * nw))?; // (nm, nh*nw)
                        let mask = coefs.dot(&proto).into_shape((nh, nw, 1))?; // (nh, nw, n)

                        // build image from ndarray
                        let mask_im: ImageBuffer<image::Luma<_>, Vec<f32>> =
                            match ImageBuffer::from_raw(nw as u32, nh as u32, mask.into_raw_vec()) {
                                Some(image) => image,
                                None => panic!("can not create image from ndarray"),
                            };
                        let mut mask_im = image::DynamicImage::from(mask_im); // -> dyn

                        // rescale masks
                        let (_, w_mask, h_mask) =
                            self.scale_wh(width_original, height_original, nw as f32, nh as f32);
                        let mask_cropped = mask_im.crop(0, 0, w_mask as u32, h_mask as u32);
                        let mask_original = mask_cropped.resize_exact(
                            // resize_to_fill
                            width_original as u32,
                            height_original as u32,
                            match self.task {
                                YOLOTask::Segment => image::imageops::FilterType::CatmullRom,
                                _ => image::imageops::FilterType::Triangle,
                            },
                        );

                        // crop-mask with bbox
                        let mut mask_original_cropped = mask_original.into_luma8();
                        for y in 0..height_original as usize {
                            for x in 0..width_original as usize {
                                if x < elem.0.xmin() as usize
                                    || x > elem.0.xmax() as usize
                                    || y < elem.0.ymin() as usize
                                    || y > elem.0.ymax() as usize
                                {
                                    mask_original_cropped.put_pixel(
                                        x as u32,
                                        y as u32,
                                        image::Luma([0u8]),
                                    );
                                }
                            }
                        }
                        y_masks.push(mask_original_cropped.into_raw());
                    }
                    y_bboxes.push(elem.0);
                }

                // save each result
                let y = YOLOResult {
                    probs: None,
                    bboxes: if !y_bboxes.is_empty() {
                        Some(y_bboxes)
                    } else {
                        None
                    },
                    keypoints: if !y_kpts.is_empty() {
                        Some(y_kpts)
                    } else {
                        None
                    },
                    masks: if !y_masks.is_empty() {
                        Some(y_masks)
                    } else {
                        None
                    },
                };
                ys.push(y);
            }

            Ok(ys)
        }
    }


}
