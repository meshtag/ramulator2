; ModuleID = 'test/attention_pim.ll'
source_filename = "test/attention_pim.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

@.str = private unnamed_addr constant [14 x i8] c"pim_trace.txt\00", align 1
@Q = internal global [64 x [64 x float]] zeroinitializer, align 4
@K = internal global [64 x [64 x float]] zeroinitializer, align 4
@V = internal global [64 x [64 x float]] zeroinitializer, align 4
@S = internal global [64 x [64 x float]] zeroinitializer, align 4
@O = internal global [64 x [64 x float]] zeroinitializer, align 4

; Function Attrs: noinline nounwind ssp uwtable(sync)
define i32 @main() local_unnamed_addr #0 {
  %1 = alloca [2 x i32], align 8
  %2 = alloca [2 x i32], align 8
  tail call void @pim_init(ptr noundef nonnull @.str) #5
  store i64 274877907008, ptr %1, align 8
  store i64 274877907008, ptr %2, align 8
  %3 = call i32 @pim_register_tensor(ptr noundef nonnull @Q, ptr noundef nonnull %1, i32 noundef 2, i32 noundef 4, i32 noundef 1) #5
  %4 = call i32 @pim_register_tensor(ptr noundef nonnull @K, ptr noundef nonnull %1, i32 noundef 2, i32 noundef 4, i32 noundef 0) #5
  %5 = call i32 @pim_register_tensor(ptr noundef nonnull @V, ptr noundef nonnull %1, i32 noundef 2, i32 noundef 4, i32 noundef 0) #5
  %6 = call i32 @pim_register_tensor(ptr noundef nonnull @S, ptr noundef nonnull %2, i32 noundef 2, i32 noundef 4, i32 noundef 2) #5
  %7 = call i32 @pim_register_tensor(ptr noundef nonnull @O, ptr noundef nonnull %1, i32 noundef 2, i32 noundef 4, i32 noundef 2) #5
  br label %vector.ph

vector.ph:                                        ; preds = %middle.block, %0
  %indvars.iv154 = phi i64 [ 0, %0 ], [ %indvars.iv.next155, %middle.block ]
  %8 = shl nsw i64 %indvars.iv154, 6
  %broadcast.splatinsert = insertelement <4 x i64> poison, i64 %8, i64 0
  %broadcast.splat = shufflevector <4 x i64> %broadcast.splatinsert, <4 x i64> poison, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.ind = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph ], [ %vec.ind.next, %vector.body ]
  %9 = add nuw nsw <4 x i64> %vec.ind, %broadcast.splat
  %10 = trunc <4 x i64> %9 to <4 x i32>
  %11 = urem <4 x i32> %10, <i32 97, i32 97, i32 97, i32 97>
  %12 = sitofp <4 x i32> %11 to <4 x float>
  %13 = fmul <4 x float> %12, <float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000>
  %14 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %indvars.iv154, i64 %index
  store <4 x float> %13, ptr %14, align 4
  %15 = trunc <4 x i64> %9 to <4 x i32>
  %16 = add <4 x i32> %15, <i32 31, i32 31, i32 31, i32 31>
  %17 = urem <4 x i32> %16, <i32 89, i32 89, i32 89, i32 89>
  %18 = sitofp <4 x i32> %17 to <4 x float>
  %19 = fmul <4 x float> %18, <float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000>
  %20 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %indvars.iv154, i64 %index
  store <4 x float> %19, ptr %20, align 4
  %21 = trunc <4 x i64> %9 to <4 x i32>
  %22 = add <4 x i32> %21, <i32 53, i32 53, i32 53, i32 53>
  %23 = urem <4 x i32> %22, <i32 83, i32 83, i32 83, i32 83>
  %24 = sitofp <4 x i32> %23 to <4 x float>
  %25 = fmul <4 x float> %24, <float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000>
  %26 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %indvars.iv154, i64 %index
  store <4 x float> %25, ptr %26, align 4
  %index.next = add nuw i64 %index, 4
  %vec.ind.next = add <4 x i64> %vec.ind, <i64 4, i64 4, i64 4, i64 4>
  %27 = icmp eq i64 %index.next, 64
  br i1 %27, label %middle.block, label %vector.body, !llvm.loop !6

middle.block:                                     ; preds = %vector.body
  %indvars.iv.next155 = add nuw nsw i64 %indvars.iv154, 1
  %exitcond157.not = icmp eq i64 %indvars.iv.next155, 64
  br i1 %exitcond157.not, label %28, label %vector.ph, !llvm.loop !10

28:                                               ; preds = %middle.block
  call void @pim_set_phase(i32 noundef 1) #5
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(16384) @S, i8 0, i64 16384, i1 false)
  br label %.preheader127

.preheader127:                                    ; preds = %28, %134
  %indvars.iv176 = phi i64 [ 0, %28 ], [ %indvars.iv.next177, %134 ]
  br label %.preheader126

.preheader126:                                    ; preds = %.preheader127, %132
  %indvars.iv173 = phi i64 [ 0, %.preheader127 ], [ %indvars.iv.next174, %132 ]
  br label %.preheader125

.preheader125:                                    ; preds = %.preheader126, %130
  %indvars.iv170 = phi i64 [ 0, %.preheader126 ], [ %indvars.iv.next171, %130 ]
  %29 = or disjoint i64 %indvars.iv170, 1
  %30 = or disjoint i64 %indvars.iv170, 2
  %31 = or disjoint i64 %indvars.iv170, 3
  %32 = or disjoint i64 %indvars.iv170, 4
  %33 = or disjoint i64 %indvars.iv170, 5
  %34 = or disjoint i64 %indvars.iv170, 6
  %35 = or disjoint i64 %indvars.iv170, 7
  %36 = or disjoint i64 %indvars.iv170, 8
  %37 = or disjoint i64 %indvars.iv170, 9
  %38 = or disjoint i64 %indvars.iv170, 10
  %39 = or disjoint i64 %indvars.iv170, 11
  %40 = or disjoint i64 %indvars.iv170, 12
  %41 = or disjoint i64 %indvars.iv170, 13
  %42 = or disjoint i64 %indvars.iv170, 14
  %43 = or disjoint i64 %indvars.iv170, 15
  br label %.preheader124

.preheader124:                                    ; preds = %.preheader125, %129
  %indvars.iv166 = phi i64 [ 0, %.preheader125 ], [ %indvars.iv.next167, %129 ]
  %44 = add nuw nsw i64 %indvars.iv166, %indvars.iv176
  %45 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %indvars.iv170
  %46 = load float, ptr %45, align 4
  %47 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %29
  %48 = load float, ptr %47, align 4
  %49 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %30
  %50 = load float, ptr %49, align 4
  %51 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %31
  %52 = load float, ptr %51, align 4
  %53 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %32
  %54 = load float, ptr %53, align 4
  %55 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %33
  %56 = load float, ptr %55, align 4
  %57 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %34
  %58 = load float, ptr %57, align 4
  %59 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %35
  %60 = load float, ptr %59, align 4
  %61 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %36
  %62 = load float, ptr %61, align 4
  %63 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %37
  %64 = load float, ptr %63, align 4
  %65 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %38
  %66 = load float, ptr %65, align 4
  %67 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %39
  %68 = load float, ptr %67, align 4
  %69 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %40
  %70 = load float, ptr %69, align 4
  %71 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %41
  %72 = load float, ptr %71, align 4
  %73 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %42
  %74 = load float, ptr %73, align 4
  %75 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %44, i64 %43
  %76 = load float, ptr %75, align 4
  br label %.preheader123

.preheader123:                                    ; preds = %.preheader124, %.preheader123
  %indvars.iv162 = phi i64 [ 0, %.preheader124 ], [ %indvars.iv.next163, %.preheader123 ]
  %77 = add nuw nsw i64 %indvars.iv162, %indvars.iv173
  %78 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %indvars.iv170
  %79 = load float, ptr %78, align 4
  %80 = call float @llvm.fmuladd.f32(float %46, float %79, float 0.000000e+00)
  %81 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %29
  %82 = load float, ptr %81, align 4
  %83 = call float @llvm.fmuladd.f32(float %48, float %82, float %80)
  %84 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %30
  %85 = load float, ptr %84, align 4
  %86 = call float @llvm.fmuladd.f32(float %50, float %85, float %83)
  %87 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %31
  %88 = load float, ptr %87, align 4
  %89 = call float @llvm.fmuladd.f32(float %52, float %88, float %86)
  %90 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %32
  %91 = load float, ptr %90, align 4
  %92 = call float @llvm.fmuladd.f32(float %54, float %91, float %89)
  %93 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %33
  %94 = load float, ptr %93, align 4
  %95 = call float @llvm.fmuladd.f32(float %56, float %94, float %92)
  %96 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %34
  %97 = load float, ptr %96, align 4
  %98 = call float @llvm.fmuladd.f32(float %58, float %97, float %95)
  %99 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %35
  %100 = load float, ptr %99, align 4
  %101 = call float @llvm.fmuladd.f32(float %60, float %100, float %98)
  %102 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %36
  %103 = load float, ptr %102, align 4
  %104 = call float @llvm.fmuladd.f32(float %62, float %103, float %101)
  %105 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %37
  %106 = load float, ptr %105, align 4
  %107 = call float @llvm.fmuladd.f32(float %64, float %106, float %104)
  %108 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %38
  %109 = load float, ptr %108, align 4
  %110 = call float @llvm.fmuladd.f32(float %66, float %109, float %107)
  %111 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %39
  %112 = load float, ptr %111, align 4
  %113 = call float @llvm.fmuladd.f32(float %68, float %112, float %110)
  %114 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %40
  %115 = load float, ptr %114, align 4
  %116 = call float @llvm.fmuladd.f32(float %70, float %115, float %113)
  %117 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %41
  %118 = load float, ptr %117, align 4
  %119 = call float @llvm.fmuladd.f32(float %72, float %118, float %116)
  %120 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %42
  %121 = load float, ptr %120, align 4
  %122 = call float @llvm.fmuladd.f32(float %74, float %121, float %119)
  %123 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %77, i64 %43
  %124 = load float, ptr %123, align 4
  %125 = call float @llvm.fmuladd.f32(float %76, float %124, float %122)
  %126 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %44, i64 %77
  %127 = load float, ptr %126, align 4
  %128 = fadd float %125, %127
  store float %128, ptr %126, align 4
  %indvars.iv.next163 = add nuw nsw i64 %indvars.iv162, 1
  %exitcond165.not = icmp eq i64 %indvars.iv.next163, 16
  br i1 %exitcond165.not, label %129, label %.preheader123, !llvm.loop !11

129:                                              ; preds = %.preheader123
  %indvars.iv.next167 = add nuw nsw i64 %indvars.iv166, 1
  %exitcond169.not = icmp eq i64 %indvars.iv.next167, 16
  br i1 %exitcond169.not, label %130, label %.preheader124, !llvm.loop !12

130:                                              ; preds = %129
  %indvars.iv.next171 = add nuw nsw i64 %indvars.iv170, 16
  %131 = icmp ult i64 %indvars.iv170, 48
  br i1 %131, label %.preheader125, label %132, !llvm.loop !13

132:                                              ; preds = %130
  %indvars.iv.next174 = add nuw nsw i64 %indvars.iv173, 16
  %133 = icmp ult i64 %indvars.iv173, 48
  br i1 %133, label %.preheader126, label %134, !llvm.loop !14

134:                                              ; preds = %132
  %indvars.iv.next177 = add nuw nsw i64 %indvars.iv176, 16
  %135 = icmp ult i64 %indvars.iv176, 48
  br i1 %135, label %.preheader127, label %136, !llvm.loop !15

136:                                              ; preds = %134
  call void @pim_set_phase(i32 noundef 2) #5
  br label %137

137:                                              ; preds = %136, %vector.body220
  %indvars.iv191 = phi i64 [ 0, %136 ], [ %indvars.iv.next192, %vector.body220 ]
  %138 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191
  %139 = load float, ptr %138, align 4
  %140 = fmul float %139, 1.250000e-01
  br label %147

vector.ph227:                                     ; preds = %147
  %141 = fneg float %.1
  %broadcast.splatinsert232 = insertelement <2 x float> poison, float %141, i64 0
  %broadcast.splat233 = shufflevector <2 x float> %broadcast.splatinsert232, <2 x float> poison, <2 x i32> zeroinitializer
  br label %vector.body229

vector.body229:                                   ; preds = %vector.body229, %vector.ph227
  %index230 = phi i64 [ 0, %vector.ph227 ], [ %index.next234, %vector.body229 ]
  %vec.phi = phi float [ 0.000000e+00, %vector.ph227 ], [ %145, %vector.body229 ]
  %142 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 %index230
  %wide.load231 = load <2 x float>, ptr %142, align 4
  %143 = call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %wide.load231, <2 x float> <float 1.250000e-01, float 1.250000e-01>, <2 x float> %broadcast.splat233)
  %144 = call <2 x float> @llvm.exp.v2f32(<2 x float> %143)
  store <2 x float> %144, ptr %142, align 4
  %145 = call float @llvm.vector.reduce.fadd.v2f32(float %vec.phi, <2 x float> %144)
  %index.next234 = add nuw i64 %index230, 2
  %146 = icmp eq i64 %index.next234, 64
  br i1 %146, label %vector.body220, label %vector.body229, !llvm.loop !16

147:                                              ; preds = %137, %147
  %indvars.iv179 = phi i64 [ 1, %137 ], [ %indvars.iv.next180, %147 ]
  %.0106138 = phi float [ %140, %137 ], [ %.1, %147 ]
  %148 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 %indvars.iv179
  %149 = load float, ptr %148, align 4
  %150 = fmul float %149, 1.250000e-01
  %151 = fcmp ogt float %150, %.0106138
  %.1 = select i1 %151, float %150, float %.0106138
  %indvars.iv.next180 = add nuw nsw i64 %indvars.iv179, 1
  %exitcond182.not = icmp eq i64 %indvars.iv.next180, 64
  br i1 %exitcond182.not, label %vector.ph227, label %147, !llvm.loop !17

vector.body220:                                   ; preds = %vector.body229
  %broadcast.splatinsert222 = insertelement <4 x float> poison, float %145, i64 0
  %broadcast.splat223 = shufflevector <4 x float> %broadcast.splatinsert222, <4 x float> poison, <4 x i32> zeroinitializer
  %152 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 0
  %wide.load = load <4 x float>, ptr %152, align 4
  %153 = fdiv <4 x float> %wide.load, %broadcast.splat223
  store <4 x float> %153, ptr %152, align 4
  %154 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 4
  %wide.load.1 = load <4 x float>, ptr %154, align 4
  %155 = fdiv <4 x float> %wide.load.1, %broadcast.splat223
  store <4 x float> %155, ptr %154, align 4
  %156 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 8
  %wide.load.2 = load <4 x float>, ptr %156, align 4
  %157 = fdiv <4 x float> %wide.load.2, %broadcast.splat223
  store <4 x float> %157, ptr %156, align 4
  %158 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 12
  %wide.load.3 = load <4 x float>, ptr %158, align 4
  %159 = fdiv <4 x float> %wide.load.3, %broadcast.splat223
  store <4 x float> %159, ptr %158, align 4
  %160 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 16
  %wide.load.4 = load <4 x float>, ptr %160, align 4
  %161 = fdiv <4 x float> %wide.load.4, %broadcast.splat223
  store <4 x float> %161, ptr %160, align 4
  %162 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 20
  %wide.load.5 = load <4 x float>, ptr %162, align 4
  %163 = fdiv <4 x float> %wide.load.5, %broadcast.splat223
  store <4 x float> %163, ptr %162, align 4
  %164 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 24
  %wide.load.6 = load <4 x float>, ptr %164, align 4
  %165 = fdiv <4 x float> %wide.load.6, %broadcast.splat223
  store <4 x float> %165, ptr %164, align 4
  %166 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 28
  %wide.load.7 = load <4 x float>, ptr %166, align 4
  %167 = fdiv <4 x float> %wide.load.7, %broadcast.splat223
  store <4 x float> %167, ptr %166, align 4
  %168 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 32
  %wide.load.8 = load <4 x float>, ptr %168, align 4
  %169 = fdiv <4 x float> %wide.load.8, %broadcast.splat223
  store <4 x float> %169, ptr %168, align 4
  %170 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 36
  %wide.load.9 = load <4 x float>, ptr %170, align 4
  %171 = fdiv <4 x float> %wide.load.9, %broadcast.splat223
  store <4 x float> %171, ptr %170, align 4
  %172 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 40
  %wide.load.10 = load <4 x float>, ptr %172, align 4
  %173 = fdiv <4 x float> %wide.load.10, %broadcast.splat223
  store <4 x float> %173, ptr %172, align 4
  %174 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 44
  %wide.load.11 = load <4 x float>, ptr %174, align 4
  %175 = fdiv <4 x float> %wide.load.11, %broadcast.splat223
  store <4 x float> %175, ptr %174, align 4
  %176 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 48
  %wide.load.12 = load <4 x float>, ptr %176, align 4
  %177 = fdiv <4 x float> %wide.load.12, %broadcast.splat223
  store <4 x float> %177, ptr %176, align 4
  %178 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 52
  %wide.load.13 = load <4 x float>, ptr %178, align 4
  %179 = fdiv <4 x float> %wide.load.13, %broadcast.splat223
  store <4 x float> %179, ptr %178, align 4
  %180 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 56
  %wide.load.14 = load <4 x float>, ptr %180, align 4
  %181 = fdiv <4 x float> %wide.load.14, %broadcast.splat223
  store <4 x float> %181, ptr %180, align 4
  %182 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %indvars.iv191, i64 60
  %wide.load.15 = load <4 x float>, ptr %182, align 4
  %183 = fdiv <4 x float> %wide.load.15, %broadcast.splat223
  store <4 x float> %183, ptr %182, align 4
  %indvars.iv.next192 = add nuw nsw i64 %indvars.iv191, 1
  %exitcond194.not = icmp eq i64 %indvars.iv.next192, 64
  br i1 %exitcond194.not, label %184, label %137, !llvm.loop !18

184:                                              ; preds = %vector.body220
  call void @pim_set_phase(i32 noundef 1) #5
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(16384) @O, i8 0, i64 16384, i1 false)
  br label %.preheader120

.preheader120:                                    ; preds = %184, %273
  %indvars.iv213 = phi i64 [ 0, %184 ], [ %indvars.iv.next214, %273 ]
  br label %.preheader119

.preheader119:                                    ; preds = %.preheader120, %271
  %indvars.iv210 = phi i64 [ 0, %.preheader120 ], [ %indvars.iv.next211, %271 ]
  br label %.preheader118

.preheader118:                                    ; preds = %.preheader119, %269
  %indvars.iv207 = phi i64 [ 0, %.preheader119 ], [ %indvars.iv.next208, %269 ]
  %185 = or disjoint i64 %indvars.iv207, 1
  %186 = or disjoint i64 %indvars.iv207, 2
  %187 = or disjoint i64 %indvars.iv207, 3
  %188 = or disjoint i64 %indvars.iv207, 4
  %189 = or disjoint i64 %indvars.iv207, 5
  %190 = or disjoint i64 %indvars.iv207, 6
  %191 = or disjoint i64 %indvars.iv207, 7
  %192 = or disjoint i64 %indvars.iv207, 8
  %193 = or disjoint i64 %indvars.iv207, 9
  %194 = or disjoint i64 %indvars.iv207, 10
  %195 = or disjoint i64 %indvars.iv207, 11
  %196 = or disjoint i64 %indvars.iv207, 12
  %197 = or disjoint i64 %indvars.iv207, 13
  %198 = or disjoint i64 %indvars.iv207, 14
  %199 = or disjoint i64 %indvars.iv207, 15
  br label %.preheader117

.preheader117:                                    ; preds = %.preheader118, %middle.block235
  %indvars.iv203 = phi i64 [ 0, %.preheader118 ], [ %indvars.iv.next204, %middle.block235 ]
  %200 = add nuw nsw i64 %indvars.iv203, %indvars.iv213
  %201 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %199
  %202 = load float, ptr %201, align 4
  %203 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %198
  %204 = load float, ptr %203, align 4
  %205 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %197
  %206 = load float, ptr %205, align 4
  %207 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %196
  %208 = load float, ptr %207, align 4
  %209 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %195
  %210 = load float, ptr %209, align 4
  %211 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %194
  %212 = load float, ptr %211, align 4
  %213 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %193
  %214 = load float, ptr %213, align 4
  %215 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %192
  %216 = load float, ptr %215, align 4
  %217 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %191
  %218 = load float, ptr %217, align 4
  %219 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %190
  %220 = load float, ptr %219, align 4
  %221 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %189
  %222 = load float, ptr %221, align 4
  %223 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %188
  %224 = load float, ptr %223, align 4
  %225 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %187
  %226 = load float, ptr %225, align 4
  %227 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %186
  %228 = load float, ptr %227, align 4
  %229 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %185
  %230 = load float, ptr %229, align 4
  %231 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %200, i64 %indvars.iv207
  %232 = load float, ptr %231, align 4
  %broadcast.splatinsert242 = insertelement <4 x float> poison, float %232, i64 0
  %broadcast.splat243 = shufflevector <4 x float> %broadcast.splatinsert242, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert245 = insertelement <4 x float> poison, float %230, i64 0
  %broadcast.splat246 = shufflevector <4 x float> %broadcast.splatinsert245, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert248 = insertelement <4 x float> poison, float %228, i64 0
  %broadcast.splat249 = shufflevector <4 x float> %broadcast.splatinsert248, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert251 = insertelement <4 x float> poison, float %226, i64 0
  %broadcast.splat252 = shufflevector <4 x float> %broadcast.splatinsert251, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert254 = insertelement <4 x float> poison, float %224, i64 0
  %broadcast.splat255 = shufflevector <4 x float> %broadcast.splatinsert254, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert257 = insertelement <4 x float> poison, float %222, i64 0
  %broadcast.splat258 = shufflevector <4 x float> %broadcast.splatinsert257, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert260 = insertelement <4 x float> poison, float %220, i64 0
  %broadcast.splat261 = shufflevector <4 x float> %broadcast.splatinsert260, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert263 = insertelement <4 x float> poison, float %218, i64 0
  %broadcast.splat264 = shufflevector <4 x float> %broadcast.splatinsert263, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert266 = insertelement <4 x float> poison, float %216, i64 0
  %broadcast.splat267 = shufflevector <4 x float> %broadcast.splatinsert266, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert269 = insertelement <4 x float> poison, float %214, i64 0
  %broadcast.splat270 = shufflevector <4 x float> %broadcast.splatinsert269, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert272 = insertelement <4 x float> poison, float %212, i64 0
  %broadcast.splat273 = shufflevector <4 x float> %broadcast.splatinsert272, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert275 = insertelement <4 x float> poison, float %210, i64 0
  %broadcast.splat276 = shufflevector <4 x float> %broadcast.splatinsert275, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert278 = insertelement <4 x float> poison, float %208, i64 0
  %broadcast.splat279 = shufflevector <4 x float> %broadcast.splatinsert278, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert281 = insertelement <4 x float> poison, float %206, i64 0
  %broadcast.splat282 = shufflevector <4 x float> %broadcast.splatinsert281, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert284 = insertelement <4 x float> poison, float %204, i64 0
  %broadcast.splat285 = shufflevector <4 x float> %broadcast.splatinsert284, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert287 = insertelement <4 x float> poison, float %202, i64 0
  %broadcast.splat288 = shufflevector <4 x float> %broadcast.splatinsert287, <4 x float> poison, <4 x i32> zeroinitializer
  br label %vector.body239

vector.body239:                                   ; preds = %vector.body239, %.preheader117
  %index240 = phi i64 [ 0, %.preheader117 ], [ %index.next290, %vector.body239 ]
  %233 = add nuw nsw i64 %index240, %indvars.iv210
  %234 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %indvars.iv207, i64 %233
  %wide.load241 = load <4 x float>, ptr %234, align 4
  %235 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat243, <4 x float> %wide.load241, <4 x float> zeroinitializer)
  %236 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %185, i64 %233
  %wide.load244 = load <4 x float>, ptr %236, align 4
  %237 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat246, <4 x float> %wide.load244, <4 x float> %235)
  %238 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %186, i64 %233
  %wide.load247 = load <4 x float>, ptr %238, align 4
  %239 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat249, <4 x float> %wide.load247, <4 x float> %237)
  %240 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %187, i64 %233
  %wide.load250 = load <4 x float>, ptr %240, align 4
  %241 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat252, <4 x float> %wide.load250, <4 x float> %239)
  %242 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %188, i64 %233
  %wide.load253 = load <4 x float>, ptr %242, align 4
  %243 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat255, <4 x float> %wide.load253, <4 x float> %241)
  %244 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %189, i64 %233
  %wide.load256 = load <4 x float>, ptr %244, align 4
  %245 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat258, <4 x float> %wide.load256, <4 x float> %243)
  %246 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %190, i64 %233
  %wide.load259 = load <4 x float>, ptr %246, align 4
  %247 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat261, <4 x float> %wide.load259, <4 x float> %245)
  %248 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %191, i64 %233
  %wide.load262 = load <4 x float>, ptr %248, align 4
  %249 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat264, <4 x float> %wide.load262, <4 x float> %247)
  %250 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %192, i64 %233
  %wide.load265 = load <4 x float>, ptr %250, align 4
  %251 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat267, <4 x float> %wide.load265, <4 x float> %249)
  %252 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %193, i64 %233
  %wide.load268 = load <4 x float>, ptr %252, align 4
  %253 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat270, <4 x float> %wide.load268, <4 x float> %251)
  %254 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %194, i64 %233
  %wide.load271 = load <4 x float>, ptr %254, align 4
  %255 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat273, <4 x float> %wide.load271, <4 x float> %253)
  %256 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %195, i64 %233
  %wide.load274 = load <4 x float>, ptr %256, align 4
  %257 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat276, <4 x float> %wide.load274, <4 x float> %255)
  %258 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %196, i64 %233
  %wide.load277 = load <4 x float>, ptr %258, align 4
  %259 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat279, <4 x float> %wide.load277, <4 x float> %257)
  %260 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %197, i64 %233
  %wide.load280 = load <4 x float>, ptr %260, align 4
  %261 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat282, <4 x float> %wide.load280, <4 x float> %259)
  %262 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %198, i64 %233
  %wide.load283 = load <4 x float>, ptr %262, align 4
  %263 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat285, <4 x float> %wide.load283, <4 x float> %261)
  %264 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %199, i64 %233
  %wide.load286 = load <4 x float>, ptr %264, align 4
  %265 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat288, <4 x float> %wide.load286, <4 x float> %263)
  %266 = getelementptr inbounds [64 x [64 x float]], ptr @O, i64 0, i64 %200, i64 %233
  %wide.load289 = load <4 x float>, ptr %266, align 4
  %267 = fadd <4 x float> %265, %wide.load289
  store <4 x float> %267, ptr %266, align 4
  %index.next290 = add nuw i64 %index240, 4
  %268 = icmp eq i64 %index.next290, 16
  br i1 %268, label %middle.block235, label %vector.body239, !llvm.loop !19

middle.block235:                                  ; preds = %vector.body239
  %indvars.iv.next204 = add nuw nsw i64 %indvars.iv203, 1
  %exitcond206.not = icmp eq i64 %indvars.iv.next204, 16
  br i1 %exitcond206.not, label %269, label %.preheader117, !llvm.loop !20

269:                                              ; preds = %middle.block235
  %indvars.iv.next208 = add nuw nsw i64 %indvars.iv207, 16
  %270 = icmp ult i64 %indvars.iv207, 48
  br i1 %270, label %.preheader118, label %271, !llvm.loop !21

271:                                              ; preds = %269
  %indvars.iv.next211 = add nuw nsw i64 %indvars.iv210, 16
  %272 = icmp ult i64 %indvars.iv210, 48
  br i1 %272, label %.preheader119, label %273, !llvm.loop !22

273:                                              ; preds = %271
  %indvars.iv.next214 = add nuw nsw i64 %indvars.iv213, 16
  %274 = icmp ult i64 %indvars.iv213, 48
  br i1 %274, label %.preheader120, label %275, !llvm.loop !23

275:                                              ; preds = %273
  call void @pim_set_phase(i32 noundef 0) #5
  call void @pim_finalize() #5
  ret i32 0
}

declare void @pim_init(ptr noundef) local_unnamed_addr #1

declare i32 @pim_register_tensor(ptr noundef, ptr noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #1

declare void @pim_set_phase(i32 noundef) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #3

declare void @pim_finalize() local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x float> @llvm.fmuladd.v2f32(<2 x float>, <2 x float>, <2 x float>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x float> @llvm.exp.v2f32(<2 x float>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v2f32(float, <2 x float>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #4

attributes #0 = { noinline nounwind ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+bti,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+bti,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"Apple clang version 17.0.0 (clang-1700.4.4.1)"}
!6 = distinct !{!6, !7, !8, !9}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!"llvm.loop.isvectorized", i32 1}
!9 = !{!"llvm.loop.unroll.runtime.disable"}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
!14 = distinct !{!14, !7}
!15 = distinct !{!15, !7}
!16 = distinct !{!16, !7, !8, !9}
!17 = distinct !{!17, !7}
!18 = distinct !{!18, !7}
!19 = distinct !{!19, !7, !8, !9}
!20 = distinct !{!20, !7}
!21 = distinct !{!21, !7}
!22 = distinct !{!22, !7}
!23 = distinct !{!23, !7}
