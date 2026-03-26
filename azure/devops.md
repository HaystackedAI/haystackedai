Azure DevOps Pipeline
  ├─ Stage: Build (already done)
  └─ Stage: Deploy
        └─ Azure App Service (Linux, container)



g
Git push
  ↓
Azure DevOps Pipeline
  ├─ Build Docker image
  ├─ Push to ACR
  └─ Deploy to Web App for Containers
        ↓
   FastAPI running in Azure


terraform import azurerm_resource_group.rg /subscriptions/63fd62a9-18db-4f66-af98-185756636dd5/resourceGroups/aisearch
terraform import azurerm_container_registry.acr /subscriptions/63fd62a9-18db-4f66-af98-185756636dd5/resourceGroups/aisearch/providers/Microsoft.ContainerRegistry/registries/aisearch8acr

terraform import azurerm_service_plan.asp "/subscriptions/63fd62a9-18db-4f66-af98-185756636dd5/resourceGroups/aisearch/providers/Microsoft.Web/serverFarms/ASP-aisearch-a6d3"
terraform import azurerm_linux_web_app.webapp /subscriptions/63fd62a9-18db-4f66-af98-185756636dd5/resourceGroups/aisearch/providers/Microsoft.Web/sites/haystacked

Terraform owns: infra (RG, ACR, ASP, Web App shell)
CI/CD owns: runtime (container image, tags, env vars)