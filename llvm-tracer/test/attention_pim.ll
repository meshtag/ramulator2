; ModuleID = 'test/attention_pim.c'
source_filename = "test/attention_pim.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

@.str = private unnamed_addr constant [14 x i8] c"pim_trace.txt\00", align 1
@__const.main.dims_qkv = private unnamed_addr constant [2 x i32] [i32 64, i32 64], align 4
@__const.main.dims_s = private unnamed_addr constant [2 x i32] [i32 64, i32 64], align 4
@Q = internal global [64 x [64 x float]] zeroinitializer, align 4
@K = internal global [64 x [64 x float]] zeroinitializer, align 4
@V = internal global [64 x [64 x float]] zeroinitializer, align 4
@S = internal global [64 x [64 x float]] zeroinitializer, align 4
@O = internal global [64 x [64 x float]] zeroinitializer, align 4

; Function Attrs: noinline nounwind ssp uwtable(sync)
define i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca [2 x i32], align 4
  %3 = alloca [2 x i32], align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca float, align 4
  %12 = alloca i32, align 4
  %13 = alloca float, align 4
  %14 = alloca i32, align 4
  %15 = alloca float, align 4
  %16 = alloca i32, align 4
  %17 = alloca float, align 4
  %18 = alloca float, align 4
  %19 = alloca i32, align 4
  %20 = alloca i32, align 4
  %21 = alloca i32, align 4
  %22 = alloca i32, align 4
  %23 = alloca i32, align 4
  %24 = alloca i32, align 4
  %25 = alloca i32, align 4
  %26 = alloca float, align 4
  %27 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  call void @pim_init(ptr noundef @.str)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %2, ptr align 4 @__const.main.dims_qkv, i64 8, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %3, ptr align 4 @__const.main.dims_s, i64 8, i1 false)
  %28 = getelementptr inbounds [2 x i32], ptr %2, i64 0, i64 0
  %29 = call i32 @pim_register_tensor(ptr noundef @Q, ptr noundef %28, i32 noundef 2, i32 noundef 4, i32 noundef 1)
  %30 = getelementptr inbounds [2 x i32], ptr %2, i64 0, i64 0
  %31 = call i32 @pim_register_tensor(ptr noundef @K, ptr noundef %30, i32 noundef 2, i32 noundef 4, i32 noundef 0)
  %32 = getelementptr inbounds [2 x i32], ptr %2, i64 0, i64 0
  %33 = call i32 @pim_register_tensor(ptr noundef @V, ptr noundef %32, i32 noundef 2, i32 noundef 4, i32 noundef 0)
  %34 = getelementptr inbounds [2 x i32], ptr %3, i64 0, i64 0
  %35 = call i32 @pim_register_tensor(ptr noundef @S, ptr noundef %34, i32 noundef 2, i32 noundef 4, i32 noundef 2)
  %36 = getelementptr inbounds [2 x i32], ptr %2, i64 0, i64 0
  %37 = call i32 @pim_register_tensor(ptr noundef @O, ptr noundef %36, i32 noundef 2, i32 noundef 4, i32 noundef 2)
  store i32 0, ptr %4, align 4
  br label %38

38:                                               ; preds = %91, %0
  %39 = load i32, ptr %4, align 4
  %40 = icmp slt i32 %39, 64
  br i1 %40, label %41, label %94

41:                                               ; preds = %38
  store i32 0, ptr %5, align 4
  br label %42

42:                                               ; preds = %87, %41
  %43 = load i32, ptr %5, align 4
  %44 = icmp slt i32 %43, 64
  br i1 %44, label %45, label %90

45:                                               ; preds = %42
  %46 = load i32, ptr %4, align 4
  %47 = mul nsw i32 %46, 64
  %48 = load i32, ptr %5, align 4
  %49 = add nsw i32 %47, %48
  %50 = srem i32 %49, 97
  %51 = sitofp i32 %50 to float
  %52 = fmul float 0x3F847AE140000000, %51
  %53 = load i32, ptr %4, align 4
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %54
  %56 = load i32, ptr %5, align 4
  %57 = sext i32 %56 to i64
  %58 = getelementptr inbounds [64 x float], ptr %55, i64 0, i64 %57
  store float %52, ptr %58, align 4
  %59 = load i32, ptr %4, align 4
  %60 = mul nsw i32 %59, 64
  %61 = load i32, ptr %5, align 4
  %62 = add nsw i32 %60, %61
  %63 = add nsw i32 %62, 31
  %64 = srem i32 %63, 89
  %65 = sitofp i32 %64 to float
  %66 = fmul float 0x3F847AE140000000, %65
  %67 = load i32, ptr %4, align 4
  %68 = sext i32 %67 to i64
  %69 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %68
  %70 = load i32, ptr %5, align 4
  %71 = sext i32 %70 to i64
  %72 = getelementptr inbounds [64 x float], ptr %69, i64 0, i64 %71
  store float %66, ptr %72, align 4
  %73 = load i32, ptr %4, align 4
  %74 = mul nsw i32 %73, 64
  %75 = load i32, ptr %5, align 4
  %76 = add nsw i32 %74, %75
  %77 = add nsw i32 %76, 53
  %78 = srem i32 %77, 83
  %79 = sitofp i32 %78 to float
  %80 = fmul float 0x3F847AE140000000, %79
  %81 = load i32, ptr %4, align 4
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %82
  %84 = load i32, ptr %5, align 4
  %85 = sext i32 %84 to i64
  %86 = getelementptr inbounds [64 x float], ptr %83, i64 0, i64 %85
  store float %80, ptr %86, align 4
  br label %87

87:                                               ; preds = %45
  %88 = load i32, ptr %5, align 4
  %89 = add nsw i32 %88, 1
  store i32 %89, ptr %5, align 4
  br label %42, !llvm.loop !6

90:                                               ; preds = %42
  br label %91

91:                                               ; preds = %90
  %92 = load i32, ptr %4, align 4
  %93 = add nsw i32 %92, 1
  store i32 %93, ptr %4, align 4
  br label %38, !llvm.loop !8

94:                                               ; preds = %38
  call void @pim_set_phase(i32 noundef 1)
  call void @llvm.memset.p0.i64(ptr align 4 @S, i8 0, i64 16384, i1 false)
  store i32 0, ptr %6, align 4
  br label %95

95:                                               ; preds = %176, %94
  %96 = load i32, ptr %6, align 4
  %97 = icmp slt i32 %96, 64
  br i1 %97, label %98, label %179

98:                                               ; preds = %95
  store i32 0, ptr %7, align 4
  br label %99

99:                                               ; preds = %172, %98
  %100 = load i32, ptr %7, align 4
  %101 = icmp slt i32 %100, 64
  br i1 %101, label %102, label %175

102:                                              ; preds = %99
  store i32 0, ptr %8, align 4
  br label %103

103:                                              ; preds = %168, %102
  %104 = load i32, ptr %8, align 4
  %105 = icmp slt i32 %104, 64
  br i1 %105, label %106, label %171

106:                                              ; preds = %103
  store i32 0, ptr %9, align 4
  br label %107

107:                                              ; preds = %164, %106
  %108 = load i32, ptr %9, align 4
  %109 = icmp slt i32 %108, 16
  br i1 %109, label %110, label %167

110:                                              ; preds = %107
  store i32 0, ptr %10, align 4
  br label %111

111:                                              ; preds = %160, %110
  %112 = load i32, ptr %10, align 4
  %113 = icmp slt i32 %112, 16
  br i1 %113, label %114, label %163

114:                                              ; preds = %111
  store float 0.000000e+00, ptr %11, align 4
  store i32 0, ptr %12, align 4
  br label %115

115:                                              ; preds = %143, %114
  %116 = load i32, ptr %12, align 4
  %117 = icmp slt i32 %116, 16
  br i1 %117, label %118, label %146

118:                                              ; preds = %115
  %119 = load i32, ptr %6, align 4
  %120 = load i32, ptr %9, align 4
  %121 = add nsw i32 %119, %120
  %122 = sext i32 %121 to i64
  %123 = getelementptr inbounds [64 x [64 x float]], ptr @Q, i64 0, i64 %122
  %124 = load i32, ptr %8, align 4
  %125 = load i32, ptr %12, align 4
  %126 = add nsw i32 %124, %125
  %127 = sext i32 %126 to i64
  %128 = getelementptr inbounds [64 x float], ptr %123, i64 0, i64 %127
  %129 = load float, ptr %128, align 4
  %130 = load i32, ptr %7, align 4
  %131 = load i32, ptr %10, align 4
  %132 = add nsw i32 %130, %131
  %133 = sext i32 %132 to i64
  %134 = getelementptr inbounds [64 x [64 x float]], ptr @K, i64 0, i64 %133
  %135 = load i32, ptr %8, align 4
  %136 = load i32, ptr %12, align 4
  %137 = add nsw i32 %135, %136
  %138 = sext i32 %137 to i64
  %139 = getelementptr inbounds [64 x float], ptr %134, i64 0, i64 %138
  %140 = load float, ptr %139, align 4
  %141 = load float, ptr %11, align 4
  %142 = call float @llvm.fmuladd.f32(float %129, float %140, float %141)
  store float %142, ptr %11, align 4
  br label %143

143:                                              ; preds = %118
  %144 = load i32, ptr %12, align 4
  %145 = add nsw i32 %144, 1
  store i32 %145, ptr %12, align 4
  br label %115, !llvm.loop !9

146:                                              ; preds = %115
  %147 = load float, ptr %11, align 4
  %148 = load i32, ptr %6, align 4
  %149 = load i32, ptr %9, align 4
  %150 = add nsw i32 %148, %149
  %151 = sext i32 %150 to i64
  %152 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %151
  %153 = load i32, ptr %7, align 4
  %154 = load i32, ptr %10, align 4
  %155 = add nsw i32 %153, %154
  %156 = sext i32 %155 to i64
  %157 = getelementptr inbounds [64 x float], ptr %152, i64 0, i64 %156
  %158 = load float, ptr %157, align 4
  %159 = fadd float %158, %147
  store float %159, ptr %157, align 4
  br label %160

160:                                              ; preds = %146
  %161 = load i32, ptr %10, align 4
  %162 = add nsw i32 %161, 1
  store i32 %162, ptr %10, align 4
  br label %111, !llvm.loop !10

163:                                              ; preds = %111
  br label %164

164:                                              ; preds = %163
  %165 = load i32, ptr %9, align 4
  %166 = add nsw i32 %165, 1
  store i32 %166, ptr %9, align 4
  br label %107, !llvm.loop !11

167:                                              ; preds = %107
  br label %168

168:                                              ; preds = %167
  %169 = load i32, ptr %8, align 4
  %170 = add nsw i32 %169, 16
  store i32 %170, ptr %8, align 4
  br label %103, !llvm.loop !12

171:                                              ; preds = %103
  br label %172

172:                                              ; preds = %171
  %173 = load i32, ptr %7, align 4
  %174 = add nsw i32 %173, 16
  store i32 %174, ptr %7, align 4
  br label %99, !llvm.loop !13

175:                                              ; preds = %99
  br label %176

176:                                              ; preds = %175
  %177 = load i32, ptr %6, align 4
  %178 = add nsw i32 %177, 16
  store i32 %178, ptr %6, align 4
  br label %95, !llvm.loop !14

179:                                              ; preds = %95
  call void @pim_set_phase(i32 noundef 2)
  %180 = call float @llvm.sqrt.f32(float 6.400000e+01)
  %181 = fdiv float 1.000000e+00, %180
  store float %181, ptr %13, align 4
  store i32 0, ptr %14, align 4
  br label %182

182:                                              ; preds = %268, %179
  %183 = load i32, ptr %14, align 4
  %184 = icmp slt i32 %183, 64
  br i1 %184, label %185, label %271

185:                                              ; preds = %182
  %186 = load i32, ptr %14, align 4
  %187 = sext i32 %186 to i64
  %188 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %187
  %189 = getelementptr inbounds [64 x float], ptr %188, i64 0, i64 0
  %190 = load float, ptr %189, align 4
  %191 = load float, ptr %13, align 4
  %192 = fmul float %190, %191
  store float %192, ptr %15, align 4
  store i32 1, ptr %16, align 4
  br label %193

193:                                              ; preds = %212, %185
  %194 = load i32, ptr %16, align 4
  %195 = icmp slt i32 %194, 64
  br i1 %195, label %196, label %215

196:                                              ; preds = %193
  %197 = load i32, ptr %14, align 4
  %198 = sext i32 %197 to i64
  %199 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %198
  %200 = load i32, ptr %16, align 4
  %201 = sext i32 %200 to i64
  %202 = getelementptr inbounds [64 x float], ptr %199, i64 0, i64 %201
  %203 = load float, ptr %202, align 4
  %204 = load float, ptr %13, align 4
  %205 = fmul float %203, %204
  store float %205, ptr %17, align 4
  %206 = load float, ptr %17, align 4
  %207 = load float, ptr %15, align 4
  %208 = fcmp ogt float %206, %207
  br i1 %208, label %209, label %211

209:                                              ; preds = %196
  %210 = load float, ptr %17, align 4
  store float %210, ptr %15, align 4
  br label %211

211:                                              ; preds = %209, %196
  br label %212

212:                                              ; preds = %211
  %213 = load i32, ptr %16, align 4
  %214 = add nsw i32 %213, 1
  store i32 %214, ptr %16, align 4
  br label %193, !llvm.loop !15

215:                                              ; preds = %193
  store float 0.000000e+00, ptr %18, align 4
  store i32 0, ptr %19, align 4
  br label %216

216:                                              ; preds = %247, %215
  %217 = load i32, ptr %19, align 4
  %218 = icmp slt i32 %217, 64
  br i1 %218, label %219, label %250

219:                                              ; preds = %216
  %220 = load i32, ptr %14, align 4
  %221 = sext i32 %220 to i64
  %222 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %221
  %223 = load i32, ptr %19, align 4
  %224 = sext i32 %223 to i64
  %225 = getelementptr inbounds [64 x float], ptr %222, i64 0, i64 %224
  %226 = load float, ptr %225, align 4
  %227 = load float, ptr %13, align 4
  %228 = load float, ptr %15, align 4
  %229 = fneg float %228
  %230 = call float @llvm.fmuladd.f32(float %226, float %227, float %229)
  %231 = call float @llvm.exp.f32(float %230)
  %232 = load i32, ptr %14, align 4
  %233 = sext i32 %232 to i64
  %234 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %233
  %235 = load i32, ptr %19, align 4
  %236 = sext i32 %235 to i64
  %237 = getelementptr inbounds [64 x float], ptr %234, i64 0, i64 %236
  store float %231, ptr %237, align 4
  %238 = load i32, ptr %14, align 4
  %239 = sext i32 %238 to i64
  %240 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %239
  %241 = load i32, ptr %19, align 4
  %242 = sext i32 %241 to i64
  %243 = getelementptr inbounds [64 x float], ptr %240, i64 0, i64 %242
  %244 = load float, ptr %243, align 4
  %245 = load float, ptr %18, align 4
  %246 = fadd float %245, %244
  store float %246, ptr %18, align 4
  br label %247

247:                                              ; preds = %219
  %248 = load i32, ptr %19, align 4
  %249 = add nsw i32 %248, 1
  store i32 %249, ptr %19, align 4
  br label %216, !llvm.loop !16

250:                                              ; preds = %216
  store i32 0, ptr %20, align 4
  br label %251

251:                                              ; preds = %264, %250
  %252 = load i32, ptr %20, align 4
  %253 = icmp slt i32 %252, 64
  br i1 %253, label %254, label %267

254:                                              ; preds = %251
  %255 = load float, ptr %18, align 4
  %256 = load i32, ptr %14, align 4
  %257 = sext i32 %256 to i64
  %258 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %257
  %259 = load i32, ptr %20, align 4
  %260 = sext i32 %259 to i64
  %261 = getelementptr inbounds [64 x float], ptr %258, i64 0, i64 %260
  %262 = load float, ptr %261, align 4
  %263 = fdiv float %262, %255
  store float %263, ptr %261, align 4
  br label %264

264:                                              ; preds = %254
  %265 = load i32, ptr %20, align 4
  %266 = add nsw i32 %265, 1
  store i32 %266, ptr %20, align 4
  br label %251, !llvm.loop !17

267:                                              ; preds = %251
  br label %268

268:                                              ; preds = %267
  %269 = load i32, ptr %14, align 4
  %270 = add nsw i32 %269, 1
  store i32 %270, ptr %14, align 4
  br label %182, !llvm.loop !18

271:                                              ; preds = %182
  call void @pim_set_phase(i32 noundef 1)
  call void @llvm.memset.p0.i64(ptr align 4 @O, i8 0, i64 16384, i1 false)
  store i32 0, ptr %21, align 4
  br label %272

272:                                              ; preds = %353, %271
  %273 = load i32, ptr %21, align 4
  %274 = icmp slt i32 %273, 64
  br i1 %274, label %275, label %356

275:                                              ; preds = %272
  store i32 0, ptr %22, align 4
  br label %276

276:                                              ; preds = %349, %275
  %277 = load i32, ptr %22, align 4
  %278 = icmp slt i32 %277, 64
  br i1 %278, label %279, label %352

279:                                              ; preds = %276
  store i32 0, ptr %23, align 4
  br label %280

280:                                              ; preds = %345, %279
  %281 = load i32, ptr %23, align 4
  %282 = icmp slt i32 %281, 64
  br i1 %282, label %283, label %348

283:                                              ; preds = %280
  store i32 0, ptr %24, align 4
  br label %284

284:                                              ; preds = %341, %283
  %285 = load i32, ptr %24, align 4
  %286 = icmp slt i32 %285, 16
  br i1 %286, label %287, label %344

287:                                              ; preds = %284
  store i32 0, ptr %25, align 4
  br label %288

288:                                              ; preds = %337, %287
  %289 = load i32, ptr %25, align 4
  %290 = icmp slt i32 %289, 16
  br i1 %290, label %291, label %340

291:                                              ; preds = %288
  store float 0.000000e+00, ptr %26, align 4
  store i32 0, ptr %27, align 4
  br label %292

292:                                              ; preds = %320, %291
  %293 = load i32, ptr %27, align 4
  %294 = icmp slt i32 %293, 16
  br i1 %294, label %295, label %323

295:                                              ; preds = %292
  %296 = load i32, ptr %21, align 4
  %297 = load i32, ptr %24, align 4
  %298 = add nsw i32 %296, %297
  %299 = sext i32 %298 to i64
  %300 = getelementptr inbounds [64 x [64 x float]], ptr @S, i64 0, i64 %299
  %301 = load i32, ptr %23, align 4
  %302 = load i32, ptr %27, align 4
  %303 = add nsw i32 %301, %302
  %304 = sext i32 %303 to i64
  %305 = getelementptr inbounds [64 x float], ptr %300, i64 0, i64 %304
  %306 = load float, ptr %305, align 4
  %307 = load i32, ptr %23, align 4
  %308 = load i32, ptr %27, align 4
  %309 = add nsw i32 %307, %308
  %310 = sext i32 %309 to i64
  %311 = getelementptr inbounds [64 x [64 x float]], ptr @V, i64 0, i64 %310
  %312 = load i32, ptr %22, align 4
  %313 = load i32, ptr %25, align 4
  %314 = add nsw i32 %312, %313
  %315 = sext i32 %314 to i64
  %316 = getelementptr inbounds [64 x float], ptr %311, i64 0, i64 %315
  %317 = load float, ptr %316, align 4
  %318 = load float, ptr %26, align 4
  %319 = call float @llvm.fmuladd.f32(float %306, float %317, float %318)
  store float %319, ptr %26, align 4
  br label %320

320:                                              ; preds = %295
  %321 = load i32, ptr %27, align 4
  %322 = add nsw i32 %321, 1
  store i32 %322, ptr %27, align 4
  br label %292, !llvm.loop !19

323:                                              ; preds = %292
  %324 = load float, ptr %26, align 4
  %325 = load i32, ptr %21, align 4
  %326 = load i32, ptr %24, align 4
  %327 = add nsw i32 %325, %326
  %328 = sext i32 %327 to i64
  %329 = getelementptr inbounds [64 x [64 x float]], ptr @O, i64 0, i64 %328
  %330 = load i32, ptr %22, align 4
  %331 = load i32, ptr %25, align 4
  %332 = add nsw i32 %330, %331
  %333 = sext i32 %332 to i64
  %334 = getelementptr inbounds [64 x float], ptr %329, i64 0, i64 %333
  %335 = load float, ptr %334, align 4
  %336 = fadd float %335, %324
  store float %336, ptr %334, align 4
  br label %337

337:                                              ; preds = %323
  %338 = load i32, ptr %25, align 4
  %339 = add nsw i32 %338, 1
  store i32 %339, ptr %25, align 4
  br label %288, !llvm.loop !20

340:                                              ; preds = %288
  br label %341

341:                                              ; preds = %340
  %342 = load i32, ptr %24, align 4
  %343 = add nsw i32 %342, 1
  store i32 %343, ptr %24, align 4
  br label %284, !llvm.loop !21

344:                                              ; preds = %284
  br label %345

345:                                              ; preds = %344
  %346 = load i32, ptr %23, align 4
  %347 = add nsw i32 %346, 16
  store i32 %347, ptr %23, align 4
  br label %280, !llvm.loop !22

348:                                              ; preds = %280
  br label %349

349:                                              ; preds = %348
  %350 = load i32, ptr %22, align 4
  %351 = add nsw i32 %350, 16
  store i32 %351, ptr %22, align 4
  br label %276, !llvm.loop !23

352:                                              ; preds = %276
  br label %353

353:                                              ; preds = %352
  %354 = load i32, ptr %21, align 4
  %355 = add nsw i32 %354, 16
  store i32 %355, ptr %21, align 4
  br label %272, !llvm.loop !24

356:                                              ; preds = %272
  call void @pim_set_phase(i32 noundef 0)
  call void @pim_finalize()
  ret i32 0
}

declare void @pim_init(ptr noundef) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #2

declare i32 @pim_register_tensor(ptr noundef, ptr noundef, i32 noundef, i32 noundef, i32 noundef) #1

declare void @pim_set_phase(i32 noundef) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #4

declare void @pim_finalize() #1

attributes #0 = { noinline nounwind ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+bti,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+bti,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"Apple clang version 17.0.0 (clang-1700.4.4.1)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
!14 = distinct !{!14, !7}
!15 = distinct !{!15, !7}
!16 = distinct !{!16, !7}
!17 = distinct !{!17, !7}
!18 = distinct !{!18, !7}
!19 = distinct !{!19, !7}
!20 = distinct !{!20, !7}
!21 = distinct !{!21, !7}
!22 = distinct !{!22, !7}
!23 = distinct !{!23, !7}
!24 = distinct !{!24, !7}
