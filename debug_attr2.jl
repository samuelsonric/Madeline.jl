using JuMP
import MathOptInterface as MOI
import Madeline

# Create a JuMP model with Madeline
model = Model(Madeline.Optimizer)
backend = JuMP.backend(model)

# Get the innermost optimizer directly
inner = backend.optimizer.model.optimizer

println("Before setting (direct access):")
println("  inner.settings.scaling = ", inner.settings.scaling)

# Set directly on innermost
MOI.set(inner, MOI.RawOptimizerAttribute("scaling"), false)

println("\nAfter setting on inner:")
println("  inner.settings.scaling = ", inner.settings.scaling)

# Now check via JuMP
println("  get_attribute(model) = ", get_attribute(model, "scaling"))

# The issue is that the wrappers don't forward correctly
# Let's verify by calling optimize and checking what settings are used
