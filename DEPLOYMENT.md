# Production Deployment Guide

## 🚀 **PRODUCTION DEPLOYMENT CHECKLIST**

### **Pre-Deployment Requirements**

#### **1. Environment Setup**

- [ ] Copy `.env.example` to `.env` and configure all variables
- [ ] Change default API keys and secret keys
- [ ] Set `ENVIRONMENT=production`
- [ ] Configure database connection if using persistent storage
- [ ] Set up monitoring and alerting endpoints

#### **2. Security Configuration**

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate API keys
python -c "import secrets; print('admin_' + secrets.token_urlsafe(32))"
python -c "import secrets; print('user_' + secrets.token_urlsafe(32))"
```

#### **3. Infrastructure**

- [ ] Docker and Docker Compose installed
- [ ] Kubernetes cluster ready (if using K8s)
- [ ] Load balancer configured
- [ ] SSL certificates configured
- [ ] Monitoring system (Prometheus/Grafana) ready

---

## **DEPLOYMENT OPTIONS**

### **Option 1: Docker Compose (Recommended for small-medium scale)**

```bash
# 1. Clone and setup
git clone https://github.com/Maaulln/pump_Predictor.git
cd pump_Predictor

# 2. Configure environment
cp .env.example .env
# Edit .env with production values

# 3. Build and deploy
docker-compose -f deployment/docker/docker-compose.yml up -d

# 4. Verify deployment
curl https://your-domain.com/health
```

### **Option 2: Kubernetes (Recommended for enterprise scale)**

```bash
# 1. Create namespace
kubectl create namespace pump-predictor

# 2. Configure secrets
kubectl create secret generic pump-predictor-secrets \
  --from-literal=secret-key=your_secret_key \
  --from-literal=api-key-admin=your_admin_key \
  --from-literal=api-key-user=your_user_key \
  -n pump-predictor

# 3. Deploy
kubectl apply -f deployment/kubernetes/ -n pump-predictor

# 4. Verify
kubectl get pods -n pump-predictor
kubectl get services -n pump-predictor
```

### **Option 3: Cloud Platforms**

#### **AWS ECS**

```bash
# 1. Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-west-2.amazonaws.com
docker build -t pump-predictor .
docker tag pump-predictor:latest your-account.dkr.ecr.us-west-2.amazonaws.com/pump-predictor:latest
docker push your-account.dkr.ecr.us-west-2.amazonaws.com/pump-predictor:latest

# 2. Deploy using ECS CLI or AWS Console
```

#### **Google Cloud Run**

```bash
# 1. Build and deploy
gcloud builds submit --tag gcr.io/your-project/pump-predictor
gcloud run deploy pump-predictor --image gcr.io/your-project/pump-predictor --platform managed
```

#### **Azure Container Instances**

```bash
# 1. Create resource group and deploy
az group create --name pump-predictor-rg --location eastus
az container create --resource-group pump-predictor-rg --name pump-predictor --image your-registry/pump-predictor:latest
```

---

## **MONITORING & OBSERVABILITY**

### **1. Health Checks**

```bash
# API Health
curl https://your-domain.com/health

# Detailed metrics
curl https://your-domain.com/metrics
```

### **2. Logging**

```bash
# Docker Compose
docker-compose logs -f api

# Kubernetes
kubectl logs -f deployment/pump-predictor-api -n pump-predictor
```

### **3. Performance Monitoring**

- Set up Prometheus metrics collection
- Configure Grafana dashboards
- Set up alerting rules

---

## **SECURITY HARDENING**

### **1. Network Security**

- [ ] Use HTTPS only (SSL/TLS certificates)
- [ ] Configure firewall rules
- [ ] Set up VPN for admin access
- [ ] Use private networks for internal communication

### **2. Application Security**

- [ ] Change all default credentials
- [ ] Enable rate limiting
- [ ] Configure CORS properly
- [ ] Set up API key rotation
- [ ] Enable request validation

### **3. Data Security**

- [ ] Encrypt data at rest
- [ ] Encrypt data in transit
- [ ] Set up backup and recovery
- [ ] Configure audit logging

---

## **SCALING CONFIGURATION**

### **Horizontal Scaling**

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pump-predictor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pump-predictor-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### **Load Balancing**

```nginx
# Nginx configuration
upstream pump_predictor {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://pump_predictor;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## **BACKUP & DISASTER RECOVERY**

### **1. Model Backup**

```bash
# Automated model backup
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "models_backup_$DATE.tar.gz" models/
aws s3 cp "models_backup_$DATE.tar.gz" s3://your-backup-bucket/models/
```

### **2. Configuration Backup**

```bash
# Backup configurations
kubectl get all -n pump-predictor -o yaml > pump-predictor-backup.yaml
```

### **3. Database Backup** (if applicable)

```bash
# PostgreSQL backup
pg_dump -h your-db-host -U username pump_predictor > backup.sql
```

---

## **PERFORMANCE OPTIMIZATION**

### **1. Resource Limits**

```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### **2. Caching Strategy**

- Redis for API response caching
- Model result caching
- Static file caching (CDN)

### **3. Database Optimization**

- Connection pooling
- Read replicas
- Query optimization
- Indexing strategy

---

## **TROUBLESHOOTING**

### **Common Issues**

#### **1. API Not Responding**

```bash
# Check container status
docker ps
kubectl get pods

# Check logs
docker logs container_name
kubectl logs pod_name

# Check resources
docker stats
kubectl top pods
```

#### **2. Model Loading Errors**

```bash
# Check model files
ls -la models/
# Check permissions
chmod 644 models/*
```

#### **3. Authentication Issues**

```bash
# Verify API keys
curl -H "Authorization: Bearer your_api_key" https://your-domain.com/health
```

### **Performance Issues**

- Monitor CPU/Memory usage
- Check database connections
- Review API response times
- Analyze network latency

---

## **MAINTENANCE PROCEDURES**

### **1. Rolling Updates**

```bash
# Kubernetes rolling update
kubectl set image deployment/pump-predictor-api api=new-image:tag

# Docker Compose update
docker-compose pull
docker-compose up -d
```

### **2. Model Updates**

```bash
# Update model files
# Zero-downtime model update procedure
```

### **3. Monitoring Health**

```bash
# Automated health check script
#!/bin/bash
response=$(curl -s -o /dev/null -w "%{http_code}" https://your-domain.com/health)
if [ $response != "200" ]; then
    echo "API is down! Status: $response"
    # Send alert
fi
```

---

## **SUPPORT & MAINTENANCE**

### **Contact Information**

- **Developer**: Your Name
- **Email**: your.email@domain.com
- **Emergency Contact**: +1-XXX-XXX-XXXX

### **Documentation**

- API Documentation: https://your-domain.com/docs
- Technical Documentation: README.md
- Runbooks: /docs/runbooks/

### **SLA Commitments**

- **Uptime**: 99.9%
- **Response Time**: < 200ms (95th percentile)
- **Support Response**: < 4 hours
