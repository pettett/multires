
tokei ash_renderer/ --exclude ash_renderer/shaders/spv
$total = (tokei --exclude ash_renderer/shaders/spv -o json | ConvertFrom-Json)

Write-Host $total.Rust.code + $total.GLSL.code + $total.Python.code

"ash_renderer/shaders/src " + (tokei ash_renderer/shaders/src -o json | ConvertFrom-Json).GLSL.code

$rust_locations = "ash_renderer/src/app", "ash_renderer/src/draw_pipelines" , "ash_renderer/src/gui", "ash_renderer/src/utility", "baker", "baker/src/lod", "baker/src/mesh", "common", "common_renderer", "metis", "evaluation"

foreach ($item in $rust_locations) {
	Write-Host $item + (tokei $item -o json | ConvertFrom-Json).Rust.code
}
